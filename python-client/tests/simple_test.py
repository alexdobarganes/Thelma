#!/usr/bin/env python3
"""
Simple WebSocket Test Client
============================

Quick and simple test client for the NinjaTrader WebSocket publisher.
Use this for basic connectivity and functionality testing.

Usage:
    python simple_test.py
"""

import asyncio
import json
import websockets
from datetime import datetime
import signal
import sys

class SimpleWebSocketTest:
    def __init__(self, host="192.168.1.65", port=6789):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}/"
        self.running = True
        self.message_count = 0
        self.historical_count = 0
        self.real_time_count = 0
        
    async def connect_and_listen(self):
        """Connect to WebSocket and listen for messages"""
        try:
            print(f"🔗 Connecting to {self.uri}")
            
            async with websockets.connect(self.uri) as websocket:
                print("✅ Connected successfully!")
                print("📊 Listening for messages... (Ctrl+C to stop)")
                print("-" * 60)
                
                async for message in websocket:
                    if not self.running:
                        break
                        
                    try:
                        # Handle ping/pong messages (sent as plain text)
                        if message == "ping":
                            print(f"🏓 Ping received (#{self.message_count + 1})")
                            await websocket.send("pong")
                            print("🏓 Pong sent")
                            self.message_count += 1
                            continue
                        
                        # Handle JSON messages
                        data = json.loads(message)
                        await self.handle_message(data, websocket)
                        
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON: {message[:100]}...")
                    except Exception as e:
                        print(f"❌ Error processing message: {e}")
                        
        except websockets.exceptions.ConnectionRefused:
            print("❌ Connection refused. Make sure NinjaTrader is running with the WebSocket indicator.")
        except Exception as e:
            print(f"❌ Connection error: {e}")
    
    async def handle_message(self, data, websocket=None):
        """Handle incoming message"""
        self.message_count += 1
        message_type = data.get('type', 'unknown')
        
        if message_type == "historical_start":
            count = data.get('count', 'unknown')
            print(f"📊 Historical data starting: {count} bars expected")
            
        elif message_type == "historical_bar":
            self.historical_count += 1
            symbol = data.get('symbol', 'N/A')
            close = data.get('close', 0)
            timestamp = data.get('timestamp', '')
            
            # Show progress every 100 bars
            if self.historical_count % 100 == 0:
                print(f"📈 Historical: {self.historical_count} bars | {symbol} @ {close} | {timestamp}")
                
        elif message_type == "historical_end":
            sent = data.get('sent', 0)
            print(f"✅ Historical data complete: {sent} bars sent, {self.historical_count} received")
            
        elif message_type == "tick":
            self.real_time_count += 1
            symbol = data.get('symbol', 'N/A')
            price = data.get('price', 0)
            volume = data.get('volume', 0)
            timestamp = data.get('timestamp', '')
            
            print(f"🎯 TICK #{self.real_time_count}: {symbol} @ ${price} Vol:{volume} | {timestamp}")
            
        elif message_type == "bar":
            self.real_time_count += 1
            symbol = data.get('symbol', 'N/A')
            open_price = data.get('open', 0)
            high = data.get('high', 0)
            low = data.get('low', 0)
            close = data.get('close', 0)
            volume = data.get('volume', 0)
            timestamp = data.get('timestamp', '')
            
            print(f"📊 BAR #{self.real_time_count}: {symbol} OHLC: {open_price}/{high}/{low}/{close} Vol:{volume}")
            print(f"   └─ Time: {timestamp}")
            
        else:
            print(f"❓ Unknown message type: {message_type}")
            print(f"   Data: {json.dumps(data, indent=2)}")
    
    def stop(self):
        """Stop the client"""
        self.running = False
        print(f"\n📊 Session Summary:")
        print(f"   Total messages: {self.message_count}")
        print(f"   Historical bars: {self.historical_count}")
        print(f"   Real-time updates: {self.real_time_count}")
        print("👋 Goodbye!")

async def main():
    """Main function"""
    client = SimpleWebSocketTest()
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutdown signal received...")
        client.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        await client.connect_and_listen()
    except KeyboardInterrupt:
        client.stop()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 Simple NinjaTrader WebSocket Test Client")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 