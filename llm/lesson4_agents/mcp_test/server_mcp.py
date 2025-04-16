
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("Hello Serwer")

@mcp.tool(name="hello_world")
def hello_world():
  return {"status": "OK", "message": "Witaj z serwera MCP!"}

@mcp.tool(
    name="echo_tool",
    description="Echo tool that repeats the input text three times"
)
def echo_tool(text: str) -> str:
    """
    Echo the input text three times.
    
    Args:
        text: The text to be echoed
        
    Returns:
        The input text repeated three times
    """
    return text * 3

@mcp.resource("echo://{text}")
def echo_template(text: str) -> str:
    """Echo the input text"""
    return f"Echo: {text}"


@mcp.prompt("echo_prompt")
def echo_prompt(text: str) -> str:
    return text

@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]

if __name__ == "__main__":
    mcp.run()
