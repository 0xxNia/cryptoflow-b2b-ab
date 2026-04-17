export const metadata = {
  title: "CryptoFlow Web",
  description: "Vercel-native analytics surface for CryptoFlow"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: "Inter, Arial, sans-serif",
          background: "#0b1020",
          color: "#f8fafc"
        }}
      >
        {children}
      </body>
    </html>
  );
}
