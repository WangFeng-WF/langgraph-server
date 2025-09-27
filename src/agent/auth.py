from langgraph_sdk import Auth

auth = Auth()

#1、实现多用户，要使用自定义鉴权@auth.authenticate。开发环境可以，生产环境不行，生产环境需要配置LANGGRAPH_CLOUD_LICENSE_KEY
#	2、postgresql存储上下文。开发环境不可以，生产环境可以。

@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    #print(f"headers: {headers}")
    """  {b'host': b'localhost:2024', b'connection': b'keep-alive', b'content-length': b'72', b'sec-ch-ua-platform': b'"Windows"', b'x-user-id': b'1234', b'user-agent': b'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36', b'sec-ch-ua': b'"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"', b'content-type': b'application/json', b'sec-ch-ua-mobile': b'?0', b'accept': b'*/*', b'origin': b'http://localhost:3000', b'sec-fetch-site': b'same-site', b'sec-fetch-mode': b'cors', b'sec-fetch-dest': b'empty', b'referer': b'http://localhost:3000/', b'accept-encoding': b'gzip, deflate, br, zstd', b'accept-language': b'zh-CN,zh;q=0.9', b'x-request-id': b'6fa2b2c7-4386-46ca-84d8-cebf8154f8bd'} """
    #api_key = headers.get("x-api-key")
    # 获取 x-user-id 的值
    # 由于 headers 可能是带字节类型的字典，需兼容处理
    user_id = headers.get("x-user-id") or headers.get(b"x-user-id")

    #if user_id is None:
    #    raise Auth.exceptions.HTTPException(status_code=401, detail="缺少用户ID，请URL中添加user_id参数。")
    
    if isinstance(user_id, bytes):
        user_id = user_id.decode("utf-8")
    print(f"user_id:{user_id}")    # 验证 API key 并获取用户信息
    
    return {
        "identity": user_id,  # 用户唯一标识
        "user_id": user_id,
        "email": "user@example.com",
        # 其他自定义字段
    }
""" @auth.on
async def add_owner(
    ctx: Auth.types.AuthContext,
    value: dict,
):
    # 在资源的 metadata 中添加用户 ID
    metadata = value.setdefault("metadata", {})
    user_id = getattr(ctx.user, "user_id", None) if ctx.user is not None else None
    if isinstance(user_id, bytes):
        user_id = user_id.decode("utf-8")
    metadata["user_id"] = user_id
    
    # 返回过滤器，确保用户只能看到自己的资源
    return {"user_id": user_id} """

