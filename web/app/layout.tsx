import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SAFE-MMSA · 蒙古语多模态情感分析",
  description: "状态感知驱动融合增强的蒙古语多模态情感分析系统",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh">
      <body>{children}</body>
    </html>
  );
}
