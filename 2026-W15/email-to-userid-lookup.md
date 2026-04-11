# 2026-W15

## 2026-04-10: Email → User ID Lookup via BigQuery

Found a way to look up user_id from email address using BigQuery.

### What works
```
bq query --nouse_legacy_sql \
  "SELECT user_id, normalized_email, date_joined
   FROM \`character-ai.load_core_identity.int_identity_email_accounts\`
   WHERE normalized_email = '<EMAIL>'"
```

- Table: `character-ai.load_core_identity.int_identity_email_accounts`
- Has `normalized_email` (plaintext), `user_id`, `date_joined`
- One email can map to multiple user_ids (re-registration)

### What doesn't work
- AlloyDB direct via alloydb-auth-proxy: proxy connects but queries hang (VPC issue)
- `char_db_datastream.public_auth_user`: data frozen since 2023-05-03, useless
- `stg_identity_signal_email`: only has sha256 hashes, no plaintext
- Bigtable: no email→user_id table exists

### Example results
- ziliang@character.ai → user_id 844629436 (2025-09-16)
- ziliangdotme@gmail.com → user_id 221534904 (2023-09-12)

### Also
- Saved as skill: `bq-email-to-userid`
- Cleaned up memory: removed pi1 IB details (covered by skill), pi1 node info, mi350-test (decommissioned), amd2 model details (covered by skills), k8s priority classes
