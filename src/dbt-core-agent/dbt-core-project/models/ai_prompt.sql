select
    id,
    title,
    instruction,
    creator,
    create_time,
    updater,
    update_time,
    type,
    fields,
    organization,
    user,
    inputs,
    sql_example,
    deleted,
    tenant_id
from {{ source('kotl_tool', 'ai_prompt') }}
where deleted = 0
