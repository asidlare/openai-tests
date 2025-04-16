
import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

async def run_client():
    
    server_file = "server_mcp.py"
    server_params = StdioServerParameters(
        command="python",  
        args=[server_file], 
        env=None  
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            ## TOOLS
            response = await session.list_tools()
            tool = response.tools[1]
            print("list_tools", tool.description, tool.inputSchema)
            
            response = await session.call_tool("hello_world")
            print("[call_tool]: hello_world:", response)

            response = await session.call_tool("echo_tool",  {"text": "John"})
            print("[call_tool]: echo_tool:", response)

            # RESOURCES
            response = await session.list_resources()
            print("list_resources", response)

            response = await session.list_resource_templates()
            print("list_resource_templates", response)

            response = await session.read_resource("echo://hello-world")
            print("[read_resource]", response)

            
            
            # PROMPTS
            response = await session.list_prompts()
            print("[list_prompts]", response.prompts)
            
            response = await session.get_prompt("echo_prompt",  {"text": "John"})
            print("[get_prompt]: echo_prompt", response.messages[0].role, response.messages[0].content.text)

            response = await session.get_prompt("debug_error", {"error": "Tutaj jest jakiś error!"})
            print("[get_prompt]: debug_error", response)
            
        

asyncio.run(run_client())
