# Security FAQ

## Password requirements

Passwords must be at least 12 characters and include uppercase, lowercase,
and a number. Passwords expire every 90 days for admin accounts.

## API keys

API keys must be rotated every 180 days. Store keys in a secrets manager;
never commit keys to source control.

## Data retention

Customer data is retained for 365 days after account closure, then
permanently deleted from production systems.
