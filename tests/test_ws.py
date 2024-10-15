import asyncio
import websockets

async def test_websocket():
    uri = "ws://localhost:5000"
    headers = {"Origin": "http://allowed-origin.com"}
    async with websockets.connect(uri, extra_headers=headers) as websocket:
        message = "hello world"
        await websocket.send(message)
        print(f"已发送消息: {message}")
        
        response = await websocket.recv()
        print(f"收到响应: {response}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
