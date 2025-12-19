/**
 * Better Auth API Route Handler - Edge Version
 * Located in website/api/auth/[...all].ts
 */
import { auth } from "../lib/auth.js";

export const config = {
  runtime: "edge",
};

export default async (req: Request) => {
  const url = new URL(req.url);
  console.log(`[AUTH-EDGE] ${req.method} request to ${url.pathname}`);

  try {
    const response = await auth.handler(req);
    console.log(`[AUTH-EDGE] Final Response Status: ${response.status}`);
    return response;
  } catch (error: any) {
    console.error(`[AUTH-EDGE-ERROR]`, error);
    return new Response(
      JSON.stringify({
        message: "Edge Auth Error",
        error: error.message,
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
};
