-- ***********************
-- mGBA-http
-- Version: 0.3.0
-- Lua interface for mGBA-http
-- https://github.com/nikouu/mGBA-http
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
	while not server do
		server, error = socket.bind(nil, port)
		if error then
			if error == socket.ERRORS.ADDRESS_IN_USE then
				port = port + 1
			else
				console:error(formatMessage("Bind", error, true))
				break
			end
		else
			local ok
			ok, error = server:listen()
			if error then
				server:close()
				console:error(formatMessage("Listen", error, true))
			else
				console:log("Socket Server: Listening on port " .. port)
				server:add("received", socketAccept)
			end
		end
	end
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
			-- it seems that the value must be non-empty in order to actually send back?
			-- thus the ACK message default
			local returnValue = messageRouter(message:match("^(.-)%s*$"))
			sock:send(returnValue)
		elseif error then
			-- seems to go into this SOCKETERRORAGAIN state for each call, but it seems fine.
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

	-- formattedLog("messageRouter: \n\tRaw message:" .. rawMessage .. "\n\tmessageType: " .. (messageType or "") .. "\n\tmessageValue1: " .. (messageValue1 or "") .. "\n\tmessageValue2: " .. (messageValue2 or "") .. "\n\tmessageValue3: " .. (messageValue3 or ""))

	if messageType == "mgba-http.button.tap" then manageButton(messageValue1)
	elseif messageType == "mgba-http.button.hold" then manageButton(messageValue1, messageValue2)
	elseif messageType == "memoryDomain.read8" then returnValue = emu.memory[messageValue1]:read8(tonumber(messageValue2))
	elseif messageType == "memoryDomain.read16" then returnValue = emu.memory[messageValue1]:read16(tonumber(messageValue2))
	elseif messageType == "memoryDomain.readRange" then returnValue = emu.memory[messageValue1]:readRange(tonumber(messageValue2), tonumber(messageValue3))
	elseif messageType == "mgba-http.emu.reset" then returnValue = emu:reset()
	elseif messageType == "mgba-http.emu.start" then returnValue = emu:loadStateSlot(1, 31)
	elseif (rawMessage == "<|ACK|>") then formattedLog("Connecting.")
	elseif (rawMessage ~= nil or rawMessage ~= '') then formattedLog("Unable to route raw message: " .. rawMessage)
	else formattedLog(messageType)	
	end

	returnValue = tostring(returnValue or defaultReturnValue);

	-- formattedLog("Returning: " .. returnValue)
	return returnValue;
end

-- ***********************
-- Button (Convenience abstraction)
-- ***********************

local keyEventQueue = {}

function manageButton(keyLetter, duration)
	formattedLog("Pressing button: " .. keyLetter .. " for " .. (tostring(duration) .. " ms" or "1 frame"))
    duration = duration or 0
    local key = keyValues[keyLetter]
    local startFrame = emu:currentFrame()
    local endFrame = startFrame + duration + 1

    -- Enqueue the key press and release event
    table.insert(keyEventQueue, {
        keyMask = (1 << key), -- Convert key to bitmask
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
-- Start
-- ***********************
beginSocket()