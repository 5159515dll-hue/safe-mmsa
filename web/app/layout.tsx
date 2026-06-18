import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SAFE-MMSA · 蒙古语多模态情感分析",
  description: "状态感知驱动融合增强的蒙古语多模态情感分析系统",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>{children}</body>
    </html>
  );
}
