"""The keys that open the gateway, who has them, and what each may do.

Give each person their own key, so one of them can be blocked without
locking everyone else out. The owner key (./llm owner-key) is not in this
list: it cannot be blocked, and only it may change the list, through
/owner/keys. The two keys ./llm makes, "api" and "admin", are in the list
under those names, so even they can be blocked.

The list lives in a JSON file the gateway rewrites, so a block survives
restarts. Usage (requests, last use, last address) is counted in memory and
written at most once a minute, so a busy key does not write on every request.
"""

import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path

ROLES = {"api": "sk-local", "admin": "sk-admin"}  # role: prefix of its keys
NAME = re.compile(r"[\w.-]{1,40}")
SAVE_EVERY = 60  # seconds


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_entry(key: str, role: str) -> dict:
    return {"key": key, "role": role, "blocked": False, "created": now(),
            "requests": 0, "last_used": None, "last_from": None}


class Keys:
    def __init__(self, path: Path, built_in: dict[str, str]):
        """built_in names the keys ./llm makes; each one's role is its name."""
        self.path = path
        self.built_in = set(built_in)
        try:
            self.entries: dict[str, dict] = json.loads(path.read_text())
        except FileNotFoundError:
            self.entries = {}
        for name, key in built_in.items():
            entry = self.entries.get(name)
            # A different key means its file was replaced to retire the old
            # one, so the new one starts unblocked.
            if entry is None or entry["key"] != key:
                self.entries[name] = new_entry(key, name)
        self.save()

    def find(self, sent: str) -> str | None:
        """The name of the key that was sent, or None."""
        found = None
        for name, entry in self.entries.items():
            # Compare against every key, so the time taken says nothing.
            if secrets.compare_digest(sent.encode(), entry["key"].encode()):
                found = name
        return found

    def used(self, name: str, address: str | None) -> None:
        entry = self.entries[name]
        entry["requests"] += 1
        entry["last_used"] = now()
        entry["last_from"] = address
        if time.monotonic() - self.saved > SAVE_EVERY:
            self.save()

    def listing(self) -> list[dict]:
        return [{"name": name, "built_in": name in self.built_in} | entry
                for name, entry in self.entries.items()]

    def create(self, name: str, role: str) -> dict:
        if not NAME.fullmatch(name):
            raise ValueError("A name is 1 to 40 letters, digits, '.', '_' or '-'.")
        if name in self.entries:
            raise ValueError(f"There is a key named {name!r} already.")
        key = f"{ROLES[role]}-{secrets.token_urlsafe(24)}"
        self.entries[name] = new_entry(key, role)
        self.save()
        return {"name": name} | self.entries[name]

    def block(self, name: str, blocked: bool) -> dict:
        self.entries[name]["blocked"] = blocked
        self.save()
        return {"name": name} | self.entries[name]

    def delete(self, name: str) -> None:
        if name in self.built_in:
            raise ValueError(f"{name!r} is made by ./llm and would come back; block it instead.")
        del self.entries[name]
        self.save()

    def save(self) -> None:
        # Written aside and swapped in, so a crash mid-write cannot lose the blocks.
        part = self.path.with_name(self.path.name + ".part")
        with open(os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as file:
            json.dump(self.entries, file, indent=2)
        os.replace(part, self.path)
        self.saved = time.monotonic()
