import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { DOCS_MANIFEST } from "@/lib/docs-manifest";

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const docId = searchParams.get("id") || "decisions";

  const item = DOCS_MANIFEST.find((d) => d.id === docId);
  if (!item) {
    return NextResponse.json({ error: "Document not found" }, { status: 404 });
  }

  // Repository root is two levels up from src/dashboard
  const repoRoot = path.resolve(process.cwd(), "../..");
  const fullPath = path.join(repoRoot, item.relativePath);

  if (!fs.existsSync(fullPath)) {
    return NextResponse.json({ error: `File not found on disk: ${item.relativePath}` }, { status: 404 });
  }

  try {
    const rawContent = fs.readFileSync(fullPath, "utf-8");
    return NextResponse.json({
      id: item.id,
      title: item.title,
      category: item.category,
      relativePath: item.relativePath,
      description: item.description,
      content: rawContent,
    });
  } catch (e) {
    return NextResponse.json({ error: `Error reading file: ${e}` }, { status: 500 });
  }
}
