import { AssessmentScreen } from "@/components/AssessmentScreen";

export default async function AssessmentPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <AssessmentScreen id={id} />;
}
