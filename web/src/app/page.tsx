import { redirect } from "next/navigation";

// The site's root is just a router. Auth state lives in a server-set cookie
// (`ls_session`); /login decides whether to bounce on to /app.
export default function Home() {
  redirect("/login");
}
