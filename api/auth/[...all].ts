import { auth } from "../../website/src/lib/auth";

export const config = {
  runtime: "edge",
};

export default async (req: Request) => {
  return auth.handler(req);
};
