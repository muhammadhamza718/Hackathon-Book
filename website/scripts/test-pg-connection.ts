import "dotenv/config";
import { Pool } from "pg";

console.log("Testing connection...");
const connectionString = process.env.DATABASE_URL;

if (!connectionString) {
  console.error("❌ DATABASE_URL is not defined in environment variables.");
  process.exit(1);
}

console.log("DATABASE_URL found (length: " + connectionString.length + ")");

const pool = new Pool({
  connectionString,
  ssl: true,
});

pool
  .connect()
  .then((client) => {
    console.log("✅ Successfully connected to PostgreSQL!");
    return client.query("SELECT NOW()").then((res) => {
      console.log("Time from DB:", res.rows[0].now);
      client.release();
      pool.end();
      process.exit(0);
    });
  })
  .catch((err) => {
    console.error("❌ Connection failed:", err);
    pool.end();
    process.exit(1);
  });
