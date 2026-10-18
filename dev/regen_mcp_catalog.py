import asyncio
import json

from mcp_email_server import app as app_module


async def main() -> dict:
    return {
        "tools": [t.model_dump(mode="json", exclude_none=True) for t in await app_module.mcp.list_tools()],
        "resources": [r.model_dump(mode="json", exclude_none=True) for r in await app_module.mcp.list_resources()],
        "resource_templates": [
            t.model_dump(mode="json", exclude_none=True) for t in await app_module.mcp.list_resource_templates()
        ],
        "prompts": [p.model_dump(mode="json", exclude_none=True) for p in await app_module.mcp.list_prompts()],
    }


actual = asyncio.run(main())
with open("tests/fixtures/mcp_catalog.json") as fh:
    old = json.load(fh)
old_tools = {t["name"]: t for t in old["tools"]}
for t in actual["tools"]:
    o = old_tools.get(t["name"])
    if o is None:
        print("NEW TOOL:", t["name"])
    else:
        new_props = set(t["inputSchema"]["properties"]) - set(o["inputSchema"]["properties"])
        if new_props:
            print(t["name"], "new props:", sorted(new_props))
        if t["description"] != o["description"]:
            print(t["name"], "description changed")
print("tools old/new:", len(old["tools"]), len(actual["tools"]))
print("resources equal:", old["resources"] == actual["resources"])
print("resource_templates equal:", old["resource_templates"] == actual["resource_templates"])
print("prompts equal:", old["prompts"] == actual["prompts"])
with open("tests/fixtures/mcp_catalog.json", "w", encoding="utf-8") as fh:
    json.dump(actual, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
print("fixture regenerated")
