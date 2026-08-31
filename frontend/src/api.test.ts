import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api, hasApiToken, setApiToken } from "./api";

function mockFetch(response: Partial<Response> & { json?: () => unknown }) {
  const spy = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    statusText: "OK",
    json: async () => ({}),
    ...response,
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}

function lastInit(spy: ReturnType<typeof vi.fn>): RequestInit {
  return spy.mock.calls[0][1] as RequestInit;
}

function headerOf(spy: ReturnType<typeof vi.fn>, name: string): string | undefined {
  return (lastInit(spy).headers as Record<string, string>)[name];
}

describe("api token 存储", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("默认没有 token", () => {
    expect(hasApiToken()).toBe(false);
  });

  it("设置后可读取，清除后消失", () => {
    setApiToken("abc");
    expect(hasApiToken()).toBe(true);

    setApiToken(null);
    expect(hasApiToken()).toBe(false);
  });

  it("不写入 localStorage，关闭标签页即失效", () => {
    setApiToken("abc");

    expect(localStorage.getItem("masp.apiToken")).toBeNull();
  });
});

describe("请求头", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("未配置 token 时不带 Authorization，保持零配置演示行为", async () => {
    const spy = mockFetch({});

    await api.health();

    expect(headerOf(spy, "Authorization")).toBeUndefined();
    expect(headerOf(spy, "Content-Type")).toBe("application/json");
  });

  it("配置 token 后带上 Bearer 头", async () => {
    setApiToken("secret-token");
    const spy = mockFetch({});

    await api.health();

    expect(headerOf(spy, "Authorization")).toBe("Bearer secret-token");
  });

  it("调用方传入的头不会被 token 覆盖", async () => {
    setApiToken("secret-token");
    const spy = mockFetch({});

    await api.createAgentRun(
      "测试目标",
      "interactive-multi-fleet",
      "conversation-1",
      "idem-1",
    );

    expect(headerOf(spy, "Idempotency-Key")).toBe("idem-1");
    expect(headerOf(spy, "Authorization")).toBe("Bearer secret-token");
  });
});

describe("错误处理", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("401 给出可操作的指引而不是原始 detail", async () => {
    mockFetch({
      ok: false,
      status: 401,
      statusText: "Unauthorized",
      json: async () => ({ detail: "缺少或无效的 Authorization" }),
    });

    await expect(api.health()).rejects.toThrow(/API token/);
  });

  it("其他错误沿用后端 detail", async () => {
    mockFetch({
      ok: false,
      status: 409,
      statusText: "Conflict",
      json: async () => ({ detail: "世界版本已变化" }),
    });

    await expect(api.health()).rejects.toThrow("世界版本已变化");
  });

  it("响应体不是 JSON 时回退到 statusText", async () => {
    mockFetch({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => {
        throw new Error("not json");
      },
    });

    await expect(api.health()).rejects.toThrow("Internal Server Error");
  });
});
