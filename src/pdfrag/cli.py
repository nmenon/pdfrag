#!/usr/bin/env python3
# ABOUTME: Command-line interface for discovering and invoking MCP server tools.
# ABOUTME: Supports both interactive and scripting workflows with flexible configuration.

"""
MCP CLI - Command-line interface for interacting with MCP servers

A tool for discovering and invoking tools from Model Context Protocol (MCP) servers.
Supports both interactive and scripting workflows with flexible configuration.
"""

import sys
import os
import json
import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPConfig:
    """Configuration for MCP server connection."""

    def __init__(self, config_file: Optional[str] = None, **overrides):
        """
        Load configuration from file and apply command-line overrides.

        Args:
            config_file: Path to JSON config file (default: ./server-config.json)
            **overrides: Command-line overrides for config values
        """
        self.config = self._load_config(config_file)
        self._apply_overrides(overrides)

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if config_file is None:
            config_file = "./server-config.json"

        config_path = Path(config_file)
        if not config_path.exists():
            return {}

        try:
            with open(config_path) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to load config file: {e}", file=sys.stderr)
            sys.exit(1)

    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply command-line overrides to configuration."""
        for key, value in overrides.items():
            if value is not None:
                self.config[key] = value

    def get_server_params(self) -> StdioServerParameters:
        """Get MCP server parameters from configuration."""
        if "server_path" not in self.config:
            print("Error: server_path not specified in config or command line", file=sys.stderr)
            sys.exit(1)

        server_path = self.config["server_path"]
        server_args = self.config.get("server_args", [])

        # Build command - use the same Python interpreter as the current process
        command = sys.executable
        args = [server_path] + server_args

        return StdioServerParameters(
            command=command,
            args=args,
            env=None
        )


class MCPClient:
    """Client for interacting with MCP servers."""

    def __init__(self, config: MCPConfig):
        """
        Initialize MCP client.

        Args:
            config: MCPConfig instance with server configuration
        """
        self.config = config
        self.server_params = config.get_server_params()

    @asynccontextmanager
    async def connect(self):
        """Connect to MCP server and yield session."""
        # Suppress server logging by redirecting stderr to devnull
        devnull = open(os.devnull, 'w')
        try:
            async with stdio_client(self.server_params, errlog=devnull) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        finally:
            devnull.close()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools from the MCP server.

        Returns:
            List of tool dictionaries with name, description, and input schema
        """
        async with self.connect() as session:
            response = await session.list_tools()
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema
                }
                for tool in response.tools
            ]

    async def describe_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific tool.

        Args:
            tool_name: Name of the tool to describe

        Returns:
            Tool dictionary with details, or None if not found
        """
        tools = await self.list_tools()
        for tool in tools:
            if tool["name"] == tool_name:
                return tool
        return None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments to pass to the tool

        Returns:
            Tool result
        """
        async with self.connect() as session:
            # Wrap arguments in "params" object as expected by MCP tools
            wrapped_args = {"params": arguments}
            result = await session.call_tool(tool_name, arguments=wrapped_args)
            return result


class ParameterParser:
    """Parser for command-line parameter syntax."""

    @staticmethod
    def parse_params(param_args: List[str]) -> Dict[str, Any]:
        """
        Parse parameter arguments in the form key=value or key=@file.

        Args:
            param_args: List of parameter strings

        Returns:
            Dictionary of parsed parameters
        """
        params = {}

        for param in param_args:
            if "=" not in param:
                print(f"Error: Invalid parameter format '{param}'. Expected key=value", file=sys.stderr)
                sys.exit(1)

            key, value = param.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Handle file reference (@filename)
            if value.startswith("@"):
                filepath = value[1:]
                params[key] = ParameterParser._load_file(filepath)
            else:
                # Try to parse as JSON, fallback to string
                params[key] = ParameterParser._parse_value(value)

        return params

    @staticmethod
    def _load_file(filepath: str) -> Any:
        """
        Load content from file (JSON or raw content).

        Args:
            filepath: Path to file to load

        Returns:
            Parsed JSON or file content as string
        """
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            sys.exit(1)

        # Try JSON first for .json files
        if path.suffix.lower() == ".json":
            try:
                with open(path) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass  # Fall back to reading as string

        # For non-JSON or failed JSON parse, read as string
        # (MCP server will handle binary files appropriately)
        try:
            with open(path, 'r') as f:
                return f.read()
        except UnicodeDecodeError:
            # Binary file - read as bytes and convert to string
            # The server will need to handle this appropriately
            with open(path, 'rb') as f:
                content = f.read()
                # Return the absolute path for binary files
                # The server should handle file paths
                return str(path.absolute())

    @staticmethod
    def _parse_value(value: str) -> Any:
        """
        Parse a value string, attempting JSON parsing first.

        Args:
            value: Value string to parse

        Returns:
            Parsed value (bool, int, float, string, etc.)
        """
        # Try boolean
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

        # Try number
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Return as string
        return value


class OutputFormatter:
    """Formatter for CLI output."""

    @staticmethod
    def format_tools_list(tools: List[Dict[str, Any]], as_json: bool = False) -> str:
        """
        Format list of tools for display.

        Args:
            tools: List of tool dictionaries
            as_json: If True, output as JSON

        Returns:
            Formatted string
        """
        if as_json:
            return json.dumps({"count": len(tools), "tools": tools}, indent=2)

        if not tools:
            return "No tools available."

        output = [f"Available Tools ({len(tools)}):\n"]
        for tool in tools:
            output.append(f"  {tool['name']}")
            if tool['description']:
                # Extract only the first line/sentence (short description)
                short_desc = tool['description'].split('\n')[0].strip()
                output.append(f"    {short_desc}")
            output.append("")

        return "\n".join(output)

    @staticmethod
    def format_tool_description(tool: Optional[Dict[str, Any]], as_json: bool = False) -> str:
        """
        Format tool description for display.

        Args:
            tool: Tool dictionary or None
            as_json: If True, output as JSON

        Returns:
            Formatted string
        """
        if tool is None:
            return json.dumps({"error": "Tool not found"}) if as_json else "Error: Tool not found"

        if as_json:
            return json.dumps(tool, indent=2)

        output = [f"Tool: {tool['name']}\n"]

        if tool['description']:
            output.append(f"Description: {tool['description']}\n")

        output.append("Parameters:")
        schema = tool.get('inputSchema', {})
        properties = schema.get('properties', {})
        required = schema.get('required', [])

        if not properties:
            output.append("  (none)")
        else:
            for param_name, param_info in properties.items():
                req_marker = " (required)" if param_name in required else ""
                param_type = param_info.get('type', 'any')
                param_desc = param_info.get('description', '')
                output.append(f"  {param_name}: {param_type}{req_marker}")
                if param_desc:
                    output.append(f"    {param_desc}")

        return "\n".join(output)

    @staticmethod
    def format_tool_result(result: Any, as_json: bool = False) -> str:
        """
        Format tool call result for display.

        Args:
            result: Tool result
            as_json: If True, output as JSON

        Returns:
            Formatted string
        """
        if as_json:
            # Extract content from result
            if hasattr(result, 'content'):
                content_list = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        content_list.append({"type": "text", "text": item.text})
                    else:
                        content_list.append({"type": "unknown", "data": str(item)})
                return json.dumps({"content": content_list}, indent=2)
            return json.dumps({"result": str(result)}, indent=2)

        # Human-readable format
        if hasattr(result, 'content'):
            output = []
            for item in result.content:
                if hasattr(item, 'text'):
                    text = item.text
                    # Try to parse as JSON and format nicely
                    try:
                        parsed = json.loads(text)
                        formatted = OutputFormatter._format_json_human(parsed)
                        output.append(formatted)
                    except (json.JSONDecodeError, ValueError):
                        # Not JSON, just append the text as-is
                        output.append(text)
                else:
                    output.append(str(item))
            return "\n".join(output)

        return str(result)

    @staticmethod
    def _format_json_human(data: Any, indent: int = 0) -> str:
        """
        Format JSON data in a human-readable way.

        Args:
            data: JSON data (dict, list, or primitive)
            indent: Current indentation level

        Returns:
            Formatted string
        """
        prefix = "  " * indent

        if isinstance(data, dict):
            if not data:
                return "{}"

            # Special handling for common response patterns
            if "status" in data:
                lines = []
                status = data.get("status", "")

                # Show status prominently
                if status == "success":
                    lines.append(f"✓ Success")
                elif status == "error":
                    lines.append(f"✗ Error")
                elif status == "already_exists":
                    lines.append(f"⚠ Already Exists")
                elif status == "not_found":
                    lines.append(f"✗ Not Found")
                else:
                    lines.append(f"Status: {status}")

                # Show message if present
                if "message" in data:
                    lines.append(f"{data['message']}")

                # Show other fields
                for key, value in data.items():
                    if key not in ["status", "message"]:
                        if isinstance(value, (dict, list)):
                            lines.append(f"\n{key}:")
                            lines.append(OutputFormatter._format_json_human(value, indent + 1))
                        else:
                            lines.append(f"{key}: {value}")

                return "\n".join(lines)

            # Default dict formatting
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(OutputFormatter._format_json_human(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
            return "\n".join(lines)

        elif isinstance(data, list):
            if not data:
                return "[]"
            lines = []
            for i, item in enumerate(data, 1):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}[{i}]")
                    lines.append(OutputFormatter._format_json_human(item, indent + 1))
                else:
                    lines.append(f"{prefix}- {item}")
            return "\n".join(lines)

        else:
            return f"{prefix}{data}"


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="mcp-cli",
        description="Command-line interface for interacting with MCP servers"
    )

    parser.add_argument(
        "--server",
        help="Path to server config file (default: ./server-config.json)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # list-tools command
    subparsers.add_parser(
        "list-tools",
        help="List all available tools from the MCP server"
    )

    # describe command
    describe_parser = subparsers.add_parser(
        "describe",
        help="Describe a specific tool"
    )
    describe_parser.add_argument("tool_name", help="Name of the tool to describe")

    # call command
    call_parser = subparsers.add_parser(
        "call",
        help="Call a tool with parameters"
    )
    call_parser.add_argument("tool_name", help="Name of the tool to call")
    call_parser.add_argument(
        "parameters",
        nargs="*",
        help="Tool parameters in key=value or key=@file format"
    )

    return parser


async def main_async(args: argparse.Namespace) -> int:
    """
    Main async function for CLI execution.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code
    """
    try:
        # Load configuration
        config = MCPConfig(
            config_file=args.server,
            server_path=None  # Not overridable via CLI for now
        )

        # Create client
        client = MCPClient(config)

        # Execute command
        if args.command == "list-tools":
            tools = await client.list_tools()
            output = OutputFormatter.format_tools_list(tools, as_json=args.json)
            print(output)
            return 0

        elif args.command == "describe":
            tool = await client.describe_tool(args.tool_name)
            output = OutputFormatter.format_tool_description(tool, as_json=args.json)
            print(output)
            return 0 if tool else 1

        elif args.command == "call":
            params = ParameterParser.parse_params(args.parameters)
            result = await client.call_tool(args.tool_name, params)
            output = OutputFormatter.format_tool_result(result, as_json=args.json)
            print(output)
            return 0

        else:
            print("Error: No command specified. Use -h for help.", file=sys.stderr)
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cli_main() -> int:
    """Main CLI logic."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run async main
    return asyncio.run(main_async(args))


def main():
    """Entry point for pdfrag-cli command."""
    sys.exit(cli_main())


if __name__ == "__main__":
    main()
