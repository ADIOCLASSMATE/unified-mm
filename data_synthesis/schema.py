"""Shared structural pair schema; routing and prompts are versioned separately."""

def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_STRING = {"type": "string"}
_STRINGS = {"type": "array", "items": _STRING}
SCHEMA = _object({
    "image_id": _STRING, "i2t": _STRING, "t2i": _STRING,
    "observations": _object({
        "counts": {"type": "array", "items": _object({
            "entity": _STRING, "count": {"type": "integer"}})},
        "relations": _STRINGS, "visible_text": _STRINGS,
    }),
    "capabilities": _STRINGS, "uncertainties": _STRINGS,
    "usable": _object({"i2t": {"type": "boolean"}, "t2i": {"type": "boolean"}}),
})
