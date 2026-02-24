import "./globals.css";
import { Space_Grotesk } from "next/font/google";
import type { ReactNode } from "react";

const spaceGrotesk = Space_Grotesk({ subsets: ["latin"], variable: "--font-display" });

export const metadata = {
  title: "LightOn OCR Capture",
  description: "Minimal capture portal for the LightOn OCR CPU endpoint."
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className={`${spaceGrotesk.variable} bg-midnight`}>{children}</body>
    </html>
  );
}
