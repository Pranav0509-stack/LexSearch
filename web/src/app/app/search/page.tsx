"use client";
import CourtSearchPane from "../court-search-pane";
import { useRouter } from "next/navigation";

export default function SearchPage() {
  const router = useRouter();
  return (
    <CourtSearchPane
      onUseInChat={(c) => {
        // Store the case context in sessionStorage, then navigate to assistant
        sessionStorage.setItem("sanhita.caseContext", JSON.stringify(c));
        router.push("/app");
      }}
    />
  );
}
