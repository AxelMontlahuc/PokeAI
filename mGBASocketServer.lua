-- ***********************
-- mGBA Socket Server
-- Original project: https://github.com/nikouu/mGBA-http
-- ***********************

local enableLogging = true

-- ***********************
-- Sockets
-- ***********************

local server = nil
local socketList = {}
local nextID = 1
local port = 8888

function beginSocket()
	local maxAttempts = 100
	local attempts = 0
	local basePort = port

	math.randomseed(os.time() + (emu and emu:currentFrame() or 0))

	local function script_dir()
		local info = debug.getinfo(1, 'S')
		local src = info and info.source or ''
		if src:sub(1,1) == '@' then
			local path = src:sub(2)
			local dir = path:match('^(.+)[/\\][^/\\]+$')
			return dir or '.'
		end
		return '.'
	end

	local function path_join(a, b)
		local sep = package.config:sub(1,1)
		if a:sub(-1) == '/' or a:sub(-1) == '\\' then
			return a .. b
		end
		return a .. sep .. b
	end

	local function ensure_dir(path)
		local sep = package.config:sub(1,1)
		if sep == '\\' then
			os.execute('mkdir "' .. path .. '" >NUL 2>&1')
		else
			os.execute('mkdir -p "' .. path .. '" >/dev/null 2>&1')
		end
	end

	local lockBase = path_join(script_dir(), 'locks')
	ensure_dir(lockBase)

	local function try_lock(p)
		local sep = package.config:sub(1,1)
		local lockName = path_join(lockBase, string.format("port-%d.lock", p))
		local tmpName = path_join(lockBase, string.format("port-%d-%d.tmp", p, math.random(1, 1e9)))
		local f = io.open(tmpName, "w")
		if not f then return false end
		f:write("locked")
		f:close()
		local ok = os.rename(tmpName, lockName)
				if not ok then
			os.remove(tmpName)
			return false
		end
		return true, lockName
	end

	local function release_lock(lock)
		if lock then pcall(os.remove, lock) end
	end

	while attempts < maxAttempts do
		local got, lockPath = try_lock(port)
		if not got then
			port = port + 1
			attempts = attempts + 1
		else
			local s, err = socket.bind(nil, port)
			if not s then
				release_lock(lockPath)
				local msg = tostring(err or "")
				if (socket.ERRORS and err == socket.ERRORS.ADDRESS_IN_USE) or msg:lower():find("in use", 1, true) then
					port = port + 1
					attempts = attempts + 1
				else
					console:error(formatMessage("Bind", msg, true))
					return
				end
			else
				local ok, lerr = s:listen()
				if not ok then
					release_lock(lockPath)
					local lmsg = tostring(lerr or "")
					s:close()
					if (socket.ERRORS and lerr == socket.ERRORS.ADDRESS_IN_USE) or lmsg:lower():find("in use", 1, true) then
						port = port + 1
						attempts = attempts + 1
					else
						console:error(formatMessage("Listen", lmsg, true))
						return
					end
				else
					server = s
					console:log("Socket Server: Listening on port " .. port .. " (locks in '" .. lockBase .. "')")
					server:add("received", socketAccept)
					return
				end
			end
		end
	end
	console:error("Socket Server: Failed to bind after " .. maxAttempts .. " attempts starting at port " .. basePort)
end

function socketAccept()
	local sock, error = server:accept()
	if error then
		console:error(formatMessage("Accept", error, true))
		return
	end
	local id = nextID
	nextID = id + 1
	socketList[id] = sock
	sock:add("received", function() socketReceived(id) end)
	sock:add("error", function() socketError(id) end)
	formattedLog(formatMessage(id, "Connected"))
end

function socketReceived(id)
	local sock = socketList[id]
	if not sock then return end
	while true do
		local message, error = sock:receive(1024)
		if message then
			local returnValue = messageRouter(message:match("^(.-)%s*$"))
			sock:send(returnValue)
		elseif error then
			if error ~= socket.ERRORS.AGAIN then
				formattedLog("socketReceived 4")
				console:error(formatMessage(id, error, true))
				socketStop(id)
			end
			return
		end
	end
end

function socketStop(id)
	local sock = socketList[id]
	socketList[id] = nil
	sock:close()
end

function socketError(id, error)
	console:error(formatMessage(id, error, true))
	socketStop(id)
end

function formatMessage(id, msg, isError)
	local prefix = "Socket " .. id
	if isError then
		prefix = prefix .. " Error: "
	else
		prefix = prefix .. " Received: "
	end
	return prefix .. msg
end

-- ***********************
-- Message Router
-- ***********************

local keyValues = {
    ["A"] = 0,
    ["B"] = 1,
    ["Select"] = 2,
    ["Start"] = 3,
    ["Right"] = 4,
    ["Left"] = 5,
    ["Up"] = 6,
    ["Down"] = 7,
    ["R"] = 8,
    ["L"] = 9
}

function messageRouter(rawMessage)
	local parsedInput = splitStringToTable(rawMessage, ",")

	local messageType = parsedInput[1]
	local messageValue1 = parsedInput[2]
	local messageValue2 = parsedInput[3]
	local messageValue3 = parsedInput[4]

	local defaultReturnValue <const> = "<|ACK|>";

	local returnValue = defaultReturnValue;

	if messageType == "mgba-http.button.tap" then manageButton(messageValue1)
	elseif messageType == "mgba-http.button.hold" then manageButton(messageValue1, messageValue2)
	elseif messageType == "memoryDomain.read8" then returnValue = emu.memory[messageValue1]:read8(tonumber(messageValue2))
	elseif messageType == "memoryDomain.read16" then returnValue = emu.memory[messageValue1]:read16(tonumber(messageValue2))
	elseif messageType == "memoryDomain.readRange" then returnValue = emu.memory[messageValue1]:readRange(tonumber(messageValue2), tonumber(messageValue3))
	elseif messageType == "mgba-http.emu.reset" then returnValue = emu:reset()
	elseif messageType == "mgba-http.emu.start" then returnValue = emu:loadStateSlot(1, 31)
	elseif messageType == "bulk.readState" then returnValue = bulkReadState()
	elseif (rawMessage == "<|ACK|>") then formattedLog("Connecting.")
	elseif (rawMessage ~= nil or rawMessage ~= '') then formattedLog("Unable to route raw message: " .. rawMessage)
	else formattedLog(messageType)	
	end

	returnValue = tostring(returnValue or defaultReturnValue);
	return returnValue;
end

-- ***********************
-- Button
-- ***********************

local keyEventQueue = {}

function manageButton(keyLetter, duration)
	formattedLog("Pressing button: " .. keyLetter .. " for " .. (tostring(duration) .. " ms" or "1 frame"))
    duration = duration or 0
    local key = keyValues[keyLetter]
    local startFrame = emu:currentFrame()
    local endFrame = startFrame + duration + 1

    table.insert(keyEventQueue, {
        keyMask = (1 << key),
        startFrame = startFrame,
        endFrame = endFrame,
        pressed = false
    })
end

function updateKeys()
    local indexesToRemove = {}

    for index, keyEvent in ipairs(keyEventQueue) do
        if emu:currentFrame() >= keyEvent.startFrame and emu:currentFrame() <= keyEvent.endFrame and not keyEvent.pressed then
            emu:addKeys(keyEvent.keyMask)
            keyEvent.pressed = true
        elseif emu:currentFrame() > keyEvent.endFrame then
            emu:clearKeys(keyEvent.keyMask)
            table.insert(indexesToRemove, index)
        end
    end

    for _, i in ipairs(indexesToRemove) do
        table.remove(keyEventQueue, i)
    end
end

callbacks:add("frame", updateKeys)

-- ***********************
-- Utility
-- ***********************

function splitStringToTable(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={}
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        table.insert(t, str)
    end
    return t
end

function formattedLog(string)
	if enableLogging then
		local timestamp = "[" .. os.date("%X", os.time()) .. "] "
		console:log(timestamp .. string)
	end
end

-- ***********************
-- Bulk State
-- ***********************

local function readVRAM2048(addr)
	local raw = emu.memory.vram:readRange(addr, 2048)
	local words = {}
	for i = 1, 2048, 2 do
		local lo = string.byte(raw, i) or 0
		local hi = string.byte(raw, i+1) or 0
		local w = lo + hi*256
		table.insert(words, w)
	end
	return words
end

local function applyOffset32x32(words, xoff, yoff)
	local out = {}
	local tx = math.floor((xoff or 0) / 8)
	local ty = math.floor((yoff or 0) / 8)
	for r = 0, 31 do
		for c = 0, 31 do
			local sr = (r + ty) % 32
			local sc = (c + tx) % 32
			local idx = sr * 32 + sc + 1
			out[#out+1] = words[idx]
		end
	end
	return out
end

function bulkReadState()
	local vals = {}

	local function push(v)
		vals[#vals+1] = tostring(v)
	end

	for i = 1, 6 do
		local base
		if i == 1 then base = 0x02024540 elseif i == 2 then base = 0x020245A4 elseif i == 3 then base = 0x02024608 elseif i == 4 then base = 0x0202466C elseif i == 5 then base = 0x020246D0 else base = 0x02024734 end
		local level = emu.memory.wram:read8(base)
		local HP = emu.memory.wram:read16(base + 0x02)
		local maxHP = emu.memory.wram:read16(base + 0x04)
		local ATK = emu.memory.wram:read16(base + 0x06)
		local DEF = emu.memory.wram:read16(base + 0x08)
		local SPEED = emu.memory.wram:read16(base + 0x0A)
		local ATK_SPE = emu.memory.wram:read16(base + 0x0C)
		local DEF_SPE = emu.memory.wram:read16(base + 0x0E)
		push(maxHP); push(HP); push(level); push(ATK); push(DEF); push(SPEED); push(ATK_SPE); push(DEF_SPE)
	end

	do
		local enemy_max = emu.memory.wram:read16(0x02024108)
		local enemy_hp = emu.memory.wram:read16(0x02024104)
		local enemy_lvl = emu.memory.wram:read8(0x02024106)
		push(enemy_max); push(enemy_hp); push(enemy_lvl)
	end

	do
		push(emu.memory.wram:read8(0x020240A8))
		push(emu.memory.wram:read8(0x020240A9))
		push(emu.memory.wram:read8(0x020240AA))
		push(emu.memory.wram:read8(0x020240AB))
	end

	do
		push(emu.memory.wram:read16(0x020322E4))
		push(emu.memory.wram:read8(0x0203cd9c))
	end

	local xoff = emu.memory.io:read8(0x04000018)
	local yoff = emu.memory.io:read8(0x0400001A)

	local bg0 = readVRAM2048(0x0600f800)
	local bg2w = readVRAM2048(0x0600e000)
	local bg2 = applyOffset32x32(bg2w, xoff, yoff)

	for i=1,1024 do push(bg0[i]) end
	for i=1,1024 do push(bg2[i]) end

	return table.concat(vals, ",")
end

-- ***********************
-- Start
-- ***********************
beginSocket()