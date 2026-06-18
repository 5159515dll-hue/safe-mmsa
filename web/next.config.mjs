/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // 部署时由 nginx 把 /api 反代到 FastAPI(8100)，前端同源调用即可。
  // 本地开发可设 NEXT_PUBLIC_API_BASE=http://127.0.0.1:8100。
};

export default nextConfig;
