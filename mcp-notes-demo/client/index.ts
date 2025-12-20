import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const transport = new StdioClientTransport({
  command: "npx",
  args: ["ts-node", "server/index.ts"]
});

const client = new Client(
  { name: "notes-client", version: "1.0.0" },
  { capabilities: {} }
);

await client.connect(transport);

// ---------- Discover tools ----------
const tools = await client.listTools();
console.log(
  "Available tools:",
  tools.tools.map(t => t.name)
);

// ---------- Create a note ----------
await client.callTool({
  name: "create_note",
  arguments: {
    title: "MCP in TypeScript",
    content: "This note was created via MCP tools."
  }
});

// ---------- List notes ----------
const listResult = await client.callTool({
  name: "list_notes"
});

console.log("\nNotes:");
const content = listResult.content as Array<{ type: string; text: string }>;
console.log(content[0].text);
