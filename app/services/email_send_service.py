"""Async SMTP email sending service.

Sends HTML emails via Gmail or Hostinger SMTP. The app password is used for
a single send and never persisted or logged.
"""
from __future__ import annotations

import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core.exceptions import AppError
from app.schemas.api import SendRequest, SendResponse

SMTP_CONFIGS = {
    "gmail": {"hostname": "smtp.gmail.com", "port": 587, "use_tls": True, "use_ssl": False},
    "hostinger": {"hostname": "smtp.hostinger.com", "port": 465, "use_tls": False, "use_ssl": True},
}


def build_html(body: str, signature: str) -> str:
    body_html = body.replace("\r\n", "\n").replace("\n", "<br>\n")
    sig_html = signature.replace("\r\n", "\n").replace("\n", "<br>\n")
    return (
        '<html><body style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6;">\n'
        f"{body_html}\n"
        '<br><hr style="border: none; border-top: 1px solid #ccc; margin: 16px 0;">\n'
        f"{sig_html}\n"
        "</body></html>"
    )


async def send_email(request: SendRequest) -> SendResponse:
    config = SMTP_CONFIGS.get(request.smtp_provider)
    if not config:
        raise AppError(f"Unknown SMTP provider: {request.smtp_provider}", status_code=400)

    html_content = build_html(request.body, request.signature)

    msg = MIMEMultipart("alternative")
    msg["From"] = request.from_email
    msg["To"] = request.to_email
    msg["Subject"] = request.subject
    msg.attach(MIMEText(request.body + "\n\n--\n" + request.signature, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    try:
        smtp = aiosmtplib.SMTP(
            hostname=config["hostname"],
            port=config["port"],
            use_tls=config["use_ssl"],
            timeout=30,
        )
        await smtp.connect()
        if config["use_tls"]:
            await smtp.starttls()
        await smtp.login(request.from_email, request.app_password)
        await smtp.send_message(msg)
        try:
            await smtp.quit()
        except Exception:
            pass
    except aiosmtplib.SMTPAuthenticationError as exc:
        return SendResponse(
            success=False,
            message=f"Authentication failed. Check your email and app password. ({exc})",
            sent_subject=request.subject,
            sent_html=html_content,
        )
    except aiosmtplib.SMTPException as exc:
        return SendResponse(
            success=False,
            message=f"SMTP error: {exc}",
            sent_subject=request.subject,
            sent_html=html_content,
        )
    except Exception as exc:
        return SendResponse(
            success=False,
            message=f"Failed to send: {exc}",
            sent_subject=request.subject,
            sent_html=html_content,
        )

    return SendResponse(
        success=True,
        message=f"Email sent to {request.to_email}",
        sent_subject=request.subject,
        sent_html=html_content,
    )
