"""Email delivery via AWS SES (plan §3.4)."""

from typing import Optional

import boto3

from app.config import get_settings


def send_document_email(
    *,
    to_email: str,
    subject: str,
    body_text: str,
    body_html: Optional[str],
    attachment_bytes: Optional[bytes] = None,
    attachment_filename: Optional[str] = None,
) -> str:
    """Returns SES message id."""
    settings = get_settings()
    client = boto3.client("ses", region_name=settings.aws_region)

    if attachment_bytes is None:
        resp = client.send_email(
            Source=settings.ses_sender,
            Destination={"ToAddresses": [to_email]},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": body_text, "Charset": "UTF-8"},
                    **({"Html": {"Data": body_html, "Charset": "UTF-8"}} if body_html else {}),
                },
            },
        )
        return resp["MessageId"]

    # Raw email path for attachments.
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = settings.ses_sender
    msg["To"] = to_email
    body = MIMEMultipart("alternative")
    body.attach(MIMEText(body_text, "plain", "utf-8"))
    if body_html:
        body.attach(MIMEText(body_html, "html", "utf-8"))
    msg.attach(body)
    part = MIMEApplication(attachment_bytes)
    part.add_header(
        "Content-Disposition",
        "attachment",
        filename=attachment_filename or "document.pdf",
    )
    msg.attach(part)

    resp = client.send_raw_email(
        Source=settings.ses_sender,
        Destinations=[to_email],
        RawMessage={"Data": msg.as_string()},
    )
    return resp["MessageId"]
