import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import fs from "fs";
import path from "path";

type Note = {
  id: number;
  title: string;
  content: string;
};

const NOTES_FILE = path.resolve("server/notes.json");

// ---------------- Helpers ----------------
function readNotes(): Note[] {
  if (!fs.existsSync(NOTES_FILE)) return [];
  return JSON.parse(fs.readFileSync(NOTES_FILE, "utf-8")) as Note[];
}

function writeNotes(notes: Note[]) {
  fs.writeFileSync(NOTES_FILE, JSON.stringify(notes, null, 2));
}

// ---------------- MCP Server ----------------
const server = new McpServer({
  name: "notes-mcp",
  version: "1.0.0"
});

// ---------------- registerTool (✅ correct) ----------------

server.registerTool(
  "list_notes",
  {
    title: "List notes",
    description: "List all notes"
  },
  async () => {
    const notes = readNotes();

    return {
      content: [
        {
          type: "text",
          text:
            notes.length === 0
              ? "No notes found."
              : notes.map(n => `• ${n.id}: ${n.title}`).join("\n")
        }
      ]
    };
  }
);

server.registerTool(
  "read_note",
  {
    title: "Read note",
    description: "Read a note by id",
    inputSchema: z.object({
      id: z.number()
    })
  },
  async ({ id }: { id: number }) => {
    const note = readNotes().find(n => n.id === id);

    return {
      content: [
        {
          type: "text",
          text: note ? note.content : "Note not found."
        }
      ]
    };
  }
);

server.registerTool(
  "create_note",
  {
    title: "Create note",
    description: "Create a new note",
    inputSchema: z.object({
      title: z.string(),
      content: z.string()
    })
  },
  async ({ title, content }: { title: string; content: string }) => {
    const notes = readNotes();

    const newNote: Note = {
      id: Date.now(),
      title,
      content
    };

    notes.push(newNote);
    writeNotes(notes);

    return {
      content: [
        {
          type: "text",
          text: `Created note with id ${newNote.id}`
        }
      ]
    };
  }
);

// ---------------- Start server ----------------
const transport = new StdioServerTransport();
await server.connect(transport);
