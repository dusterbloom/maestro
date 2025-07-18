import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import { Provider } from 'jotai'
import '@/styles/globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Maestro Voice Assistant',
  description: 'Ultra-low latency voice assistant',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Provider>{children}</Provider>
      </body>
    </html>
  )
}