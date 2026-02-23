-- ***********************
-- mGBA Socket Server
-- Original project: https://github.com/nikouu/mGBA-http
-- ***********************

local enableLogging = true

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
	elseif messageType == "memoryDomain.read32" then returnValue = emu.memory[messageValue1]:read32(tonumber(messageValue2))
	elseif messageType == "memoryDomain.readRange" then returnValue = emu.memory[messageValue1]:readRange(tonumber(messageValue2), tonumber(messageValue3))
	elseif messageType == "mgba-http.emu.reset" then returnValue = emu:reset()
	elseif messageType == "mgba-http.emu.start" then returnValue = emu:loadStateSlot(1, 31)
	elseif messageType == "bulk.readState" then returnValue = bulkReadState()
	elseif messageType == "bulk.readBehaviorMap" then returnValue = bulkReadBehaviorMap()
	elseif (rawMessage == "<|ACK|>") then formattedLog("Connecting.")
	elseif (rawMessage ~= nil or rawMessage ~= '') then formattedLog("Unable to route raw message: " .. rawMessage)
	else formattedLog(messageType)	
	end

	returnValue = tostring(returnValue or defaultReturnValue);
	return returnValue;
end

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

-- Read a 32-bit value from WRAM (little-endian)
local function readWram32(addr)
	local lo = emu.memory.wram:read16(addr)
	local hi = emu.memory.wram:read16(addr + 2)
	return lo + hi * 65536
end

-- Read a 32-bit value from ROM/cart (little-endian)
-- mGBA uses cart0 for ROM memory, not 'rom'
local function readRom32(addr)
	-- ROM addresses are 0x08xxxxxx, but we read with offset from base
	local offset = addr - 0x08000000
	local lo = emu.memory.cart0:read16(offset)
	local hi = emu.memory.cart0:read16(offset + 2)
	return lo + hi * 65536
end

local function readRom16(addr)
	local offset = addr - 0x08000000
	return emu.memory.cart0:read16(offset)
end

local function readRom8(addr)
	local offset = addr - 0x08000000
	return emu.memory.cart0:read8(offset)
end

-- Get behavior byte for a metatile
local function getTileBehavior(tile_id, primary_ts, secondary_ts)
	local tileset_ptr, local_id
	
	if tile_id < 640 then
		tileset_ptr = primary_ts
		local_id = tile_id
	else
		tileset_ptr = secondary_ts
		local_id = tile_id - 640
	end
	
	if tileset_ptr == 0 or tileset_ptr < 0x08000000 then return 0 end
	
	-- Behavior/attributes pointer is at tileset_header + 0x10
	local attr_ptr = readRom32(tileset_ptr + 0x10)
	if attr_ptr == 0 or attr_ptr < 0x08000000 then return 0 end
	
	-- Each metatile has 4 bytes of attributes, behavior is byte 0
	local behavior_addr = attr_ptr + local_id * 4
	return readRom8(behavior_addr)
end

-- Classify behavior byte: 0=normal, 1=interactable
-- Only used to detect interactable tiles; collision bits handle solid
local function classifyBehavior(behavior)
	-- Interactable behaviors
	-- if behavior == 0x08 then return 1 end    -- Houses Door
	-- if behavior == 0x28 then return 1 end    -- Lab door
	-- if behavior == 0xAE then return 1 end	 -- General doors with a 1 offset?
	-- if behavior == 0x41 then return 1 end	 -- Random door thing
	return 0
end

-- Special tile IDs that should override behavior classification
-- Maps tile_id -> behavior value for objects not distinguishable by behavior byte
-- Returns nil when no override applies
local function classifyTileId(tile_id)
	if tile_id == 655 then return 1 end      -- Clock
	return nil  -- No override
end

-- Check if map coordinate (mx, my) is a warp destination.
-- gMapHeader is at 0x02037318 in WRAM:
--   +0x00 = mapLayout ptr  (already used)
--   +0x04 = events ptr     -> MapEvents in ROM
-- MapEvents:
--   +0x00 = objectEventCount (u8)
--   +0x01 = warpCount (u8)
--   +0x02 = coordEventCount (u8)
--   +0x03 = bgEventCount (u8)
--   +0x04 = objectEvents ptr (4)
--   +0x08 = warps ptr (4)     -> array of WarpEvent
-- WarpEvent (8 bytes): x(s16) y(s16) elevation(u8) warpId(u8) mapNum(u8) mapGroup(u8)
local function isWarpTile(mx, my)
	local events_ptr = readWram32(0x0203731C)
	if events_ptr == 0 or events_ptr < 0x08000000 then return false end

	local warp_count = readRom8(events_ptr + 0x01)
	if warp_count == 0 or warp_count > 64 then return false end

	local warps_ptr = readRom32(events_ptr + 0x08)
	if warps_ptr == 0 or warps_ptr < 0x08000000 then return false end

	for w = 0, warp_count - 1 do
		local warp_addr = warps_ptr + w * 8
		local wx = readRom16(warp_addr + 0x00)
		local wy = readRom16(warp_addr + 0x02)
		-- Convert unsigned 16-bit to signed
		if wx >= 0x8000 then wx = wx - 0x10000 end
		if wy >= 0x8000 then wy = wy - 0x10000 end
		if wx == mx and wy == my then return true end
	end
	return false
end

function bulkReadBehaviorMap()
	local vals = {}
	local function push(v)
		vals[#vals+1] = tostring(v)
	end
	
	-- Map layout pointer at 0x02037318 (WRAM)
	local layout_ptr = readWram32(0x02037318)
	
	-- Player position at 0x02037360/62
	local player_x = emu.memory.wram:read16(0x02037360)
	local player_y = emu.memory.wram:read16(0x02037362)
	push(player_x)
	push(player_y)
	
	-- Debug: print layout info to console
	console:log(string.format("Layout ptr: 0x%08X, Player: (%d, %d)", layout_ptr, player_x, player_y))
	
	-- Check if layout pointer is valid (should point to ROM 0x08xxxxxx)
	if layout_ptr == 0 or layout_ptr < 0x08000000 or layout_ptr > 0x09FFFFFF then
		-- Invalid pointer, return zeros
		console:log("Invalid layout pointer!")
		for i = 1, 121 do push(0) end
		return table.concat(vals, ",")
	end
	
	-- Read layout structure from ROM
	-- Try: width(4), height(4), border(4), tile_data(4), ts1_header(4), ts2_header(4)
	local map_width = readRom32(layout_ptr + 0x00)
	local map_height = readRom32(layout_ptr + 0x04)
	local border_ptr = readRom32(layout_ptr + 0x08)  -- border tiles pointer (skip)
	local map_data = readRom32(layout_ptr + 0x0C)    -- actual map data
	local primary_ts = readRom32(layout_ptr + 0x10)  -- primary tileset
	local secondary_ts = readRom32(layout_ptr + 0x14) -- secondary tileset
	
	console:log(string.format("  Width: %d, Height: %d", map_width, map_height))
	console:log(string.format("  Border ptr: 0x%08X (skipped)", border_ptr))
	console:log(string.format("  Map data: 0x%08X", map_data))
	console:log(string.format("  Primary TS header: 0x%08X, Secondary TS header: 0x%08X", primary_ts, secondary_ts))
	
	-- Debug: Show the behavior array pointers from each tileset header
	if primary_ts >= 0x08000000 then
		console:log(string.format("  Primary TS header contents:"))
		console:log(string.format("    +0x00: 0x%08X", readRom32(primary_ts + 0x00)))
		console:log(string.format("    +0x04: 0x%08X", readRom32(primary_ts + 0x04)))
		console:log(string.format("    +0x08: 0x%08X", readRom32(primary_ts + 0x08)))
		console:log(string.format("    +0x0C: 0x%08X", readRom32(primary_ts + 0x0C)))
		console:log(string.format("    +0x10: 0x%08X", readRom32(primary_ts + 0x10)))
		console:log(string.format("    +0x14: 0x%08X", readRom32(primary_ts + 0x14)))
		local primary_behavior = readRom32(primary_ts + 0x10)
		console:log(string.format("  Primary behavior array: 0x%08X", primary_behavior))
		-- Show first few behavior bytes
		if primary_behavior >= 0x08000000 then
			local b0 = readRom8(primary_behavior + 0)
			local b1 = readRom8(primary_behavior + 4)
			local b2 = readRom8(primary_behavior + 8)
			local b3 = readRom8(primary_behavior + 12)
			console:log(string.format("    First behaviors: %02X %02X %02X %02X", b0, b1, b2, b3))
		end
	end
	if secondary_ts >= 0x08000000 then
		local secondary_behavior = readRom32(secondary_ts + 0x10)
		console:log(string.format("  Secondary behavior array: 0x%08X", secondary_behavior))
	end
	
	-- Sanity check
	if map_width == 0 or map_width > 1000 or map_height == 0 or map_height > 1000 then
		console:log("Invalid map dimensions!")
		for i = 1, 121 do push(0) end
		return table.concat(vals, ",")
	end
	
	-- Pokemon maps have a border offset of 7 tiles
	-- Player coords are in "map space" which includes this border
	-- So we need to subtract the border when indexing into map_data
	local BORDER = 7
	local adj_x = player_x - BORDER
	local adj_y = player_y - BORDER
	
	console:log(string.format("  Adjusted pos: (%d, %d)", adj_x, adj_y))
	
	-- Debug: Print tile IDs, collision bits, and behavior bytes
	-- Tile entry format: bits 0-9 = tile ID, bits 10-11 = collision, bits 12-15 = elevation
	console:log("  Tile data (11x11 grid): [TileID:Coll:Behavior]")
	for dr = -5, 5 do
		local row_str = "    "
		for dc = -5, 5 do
			local mx = adj_x + dc
			local my = adj_y + dr
			
			if mx < 0 or my < 0 or mx >= map_width or my >= map_height then
				row_str = row_str .. " [OOB]"
			else
				local tile_addr = map_data + (my * map_width + mx) * 2
				local tile_entry = readRom16(tile_addr)
				local tile_id = tile_entry % 1024  -- bits 0-9
				local collision = math.floor(tile_entry / 1024) % 4  -- bits 10-11
				local behavior = getTileBehavior(tile_id, primary_ts, secondary_ts)
				row_str = row_str .. string.format(" [%03d:%d:%02X]", tile_id, collision, behavior)
			end
		end
		console:log(row_str)
	end
	
	-- Read 11x11 grid centered on player (using adjusted coordinates)
	for dr = -5, 5 do
		for dc = -5, 5 do
			local mx = adj_x + dc
			local my = adj_y + dr
			
			-- Bounds check against actual map dimensions
			if mx < 0 or my < 0 or mx >= map_width or my >= map_height then
				push(-1)  -- Out of bounds = solid
			else
				-- Each metatile entry is 2 bytes
				-- Bits 0-9: tile ID, bits 10-11: collision, bits 12-15: elevation
				local tile_addr = map_data + (my * map_width + mx) * 2
				local tile_entry = readRom16(tile_addr)
				local tile_id = tile_entry % 1024  -- bits 0-9
				local collision = math.floor(tile_entry / 1024) % 4  -- bits 10-11
				
				-- Check tile ID for special objects (takes highest priority)
				local id_class = classifyTileId(tile_id)
				if id_class ~= nil then
					push(id_class)  -- Special object
				elseif isWarpTile(mx, my) then
					push(1)  -- Warp/door
				else
					-- Fall back to behavior byte / collision classification
					local behavior = getTileBehavior(tile_id, primary_ts, secondary_ts)
					local classified = classifyBehavior(behavior)
					if classified == 1 then
						push(1)  -- Interactable always shows through
					elseif collision ~= 0 then
						push(-1)  -- Collision flag = solid
					else
						push(0)  -- Walkable
					end
				end
			end
		end
	end
	
	return table.concat(vals, ",")
end

beginSocket()