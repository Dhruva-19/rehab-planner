"""
src/storage/auth_utils.py

Purpose (Day 26, step 1a): the security-sensitive helpers for login. Pure
functions only -- no database access here, so each one can be tested alone.

Only the Python standard library is used (hashlib, hmac, secrets), so this adds
NO new dependency to requirements.txt.

Security rules this file follows
--------------------------------
1. Passwords are never stored. We store a salted PBKDF2-HMAC-SHA256 hash.
   PBKDF2 is deliberately slow, so guessing passwords from a stolen database
   is expensive.
2. Every password gets its own random salt, so two users with the same
   password have different hashes.
3. Comparisons use hmac.compare_digest (constant time), so response timing
   cannot leak how many characters matched.
4. Login tokens are random (secrets module, not `random`) and only their
   SHA-256 hash is stored in the database.
"""

import hashlib
import hmac
import re
import secrets

ALGORITHM = "pbkdf2_sha256"

# Work factor recommended for PBKDF2-HMAC-SHA256 by OWASP. Higher = slower for
# attackers AND for your server at login. It is stored inside each hash, so it
# can be changed later without breaking existing accounts.
PBKDF2_ITERATIONS = 600_000

USERNAME_PATTERN = re.compile(r"^[a-z0-9_]{3,30}$")
MIN_PASSWORD_LENGTH = 8


# ---------------------------------------------------------------- passwords --
def hash_password(password: str) -> str:
    """Return 'pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>'."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"{ALGORITHM}${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    True only if `password` matches `stored_hash`.
    A malformed stored hash returns False instead of raising, so a corrupt row
    can never crash the login route or be used to probe the system.
    """
    try:
        algorithm, iterations, salt_hex, hash_hex = stored_hash.split("$")
        if algorithm != ALGORITHM:
            return False
        expected = bytes.fromhex(hash_hex)
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(candidate, expected)


# -------------------------------------------------------------- validation --
def normalize_username(username: str) -> str:
    """'  Dhruva ' -> 'dhruva'. Always normalise BEFORE validating or saving."""
    return username.strip().lower()


def validate_credentials(username: str, password: str) -> str | None:
    """
    Return a human-readable error message, or None if both are acceptable.
    `username` must already be normalised.
    """
    if not USERNAME_PATTERN.match(username):
        return "Username must be 3-30 characters: letters, numbers or underscore."
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    return None


# ------------------------------------------------------------ login tokens --
def new_token() -> str:
    """A fresh random token (43 URL-safe characters). Goes in the cookie."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """SHA-256 of a token. THIS is what the database stores, never the token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    # Smoke test: run with  py -3.12 src/storage/auth_utils.py
    import time

    start = time.perf_counter()
    stored = hash_password("correct horse battery")
    print(f"hash_password took {time.perf_counter() - start:.2f}s")
    print("stored format   :", stored[:40] + "...")

    assert verify_password("correct horse battery", stored) is True
    assert verify_password("wrong password", stored) is False
    assert verify_password("anything", "not-a-valid-hash") is False
    assert hash_password("same") != hash_password("same")   # per-user salt

    assert normalize_username("  Dhruva ") == "dhruva"
    assert validate_credentials("dhruva", "longenough") is None
    assert validate_credentials("d!", "longenough") is not None
    assert validate_credentials("dhruva", "short") is not None

    token = new_token()
    assert len(hash_token(token)) == 64 and hash_token(token) != token

    print("All auth_utils checks passed.")
