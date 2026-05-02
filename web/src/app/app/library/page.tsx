"use client";
import LibraryPane from "../library-pane";
import { useRouter } from "next/navigation";

export default function LibraryPage() {
  const router = useRouter();
  return (
    <LibraryPane
      onUseInChat={(doc) => {
        sessionStorage.setItem(
          "sanhita.caseContext",
          JSON.stringify({
            case_id: `lib-${Date.now()}`,
            title: doc.title,
            body_md: `Use this ${doc.kind} as context:\n\n**${doc.title}**\n\n${doc.body_md.slice(0, 1200)}`,
            jurisdiction: "IN",
          })
        );
        router.push("/app");
      }}
    />
  );
}
