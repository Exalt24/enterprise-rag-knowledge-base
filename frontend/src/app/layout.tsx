import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Enterprise RAG Knowledge Base",
  description: "Production-ready Retrieval-Augmented Generation system with hybrid search, Redis caching, and 2-tier LLM fallback. Built with FastAPI, LangChain, Chroma, and Next.js.",
  keywords: ["RAG", "AI", "Knowledge Base", "Vector Database", "LangChain", "FastAPI", "Next.js"],
  authors: [{ name: "Daniel Alexis Cruz" }],
  openGraph: {
    title: "Enterprise RAG Knowledge Base",
    description: "Production-ready RAG system with 67.7% retrieval accuracy, hybrid search, and Redis caching",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
