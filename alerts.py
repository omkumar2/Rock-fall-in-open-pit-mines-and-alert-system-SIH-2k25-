import os
import smtplib
import logging
from typing import Optional
from email.message import EmailMessage
from datetime import datetime, timezone


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _format_probability(probability: float) -> str:
    try:
        return f"{float(probability) * 100:.2f}%"
    except Exception:
        return str(probability)


def send_email_alert(location: str, risk_level: str, probability: float) -> bool:
    """Send a rockfall alert via email using SMTP.

    Environment variables used:
    - SMTP_HOST: SMTP server host (required)
    - SMTP_PORT: SMTP server port (default: 587)
    - SMTP_USERNAME: SMTP username (optional)
    - SMTP_PASSWORD: SMTP password (optional)
    - SMTP_STARTTLS: 'true' to use STARTTLS (default: 'true')
    - SMTP_FROM: Sender email address (required)
    - SMTP_TO: Recipient email address(es), comma-separated (required)
    """

    smtp_host: Optional[str] = os.getenv("SMTP_HOST")
    smtp_port_str: str = os.getenv("SMTP_PORT", "587")
    smtp_user: Optional[str] = os.getenv("SMTP_USERNAME")
    smtp_pass: Optional[str] = os.getenv("SMTP_PASSWORD")
    use_starttls: bool = os.getenv("SMTP_STARTTLS", "true").lower() == "true"
    sender: Optional[str] = os.getenv("SMTP_FROM")
    recipients_raw: Optional[str] = os.getenv("SMTP_TO")

    if not smtp_host or not sender or not recipients_raw:
        logger.error("Email alert not sent: SMTP_HOST, SMTP_FROM, and SMTP_TO are required.")
        return False

    try:
        smtp_port: int = int(smtp_port_str)
    except ValueError:
        logger.error("Email alert not sent: SMTP_PORT must be an integer.")
        return False

    recipients = [addr.strip() for addr in recipients_raw.split(",") if addr.strip()]
    if not recipients:
        logger.error("Email alert not sent: no valid recipients in SMTP_TO.")
        return False

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    probability_text = _format_probability(probability)

    subject = "🚨 Rockfall Alert"
    body = (
        "Rockfall alert details:\n"
        f"Location: {location}\n"
        f"Risk: {risk_level}\n"
        f"Probability: {probability_text}\n"
        f"Timestamp: {timestamp}\n"
    )

    message = EmailMessage()
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    message.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.ehlo()
            if use_starttls:
                try:
                    server.starttls()
                    server.ehlo()
                except smtplib.SMTPException:
                    logger.exception("STARTTLS failed; proceeding without encryption if server allows.")
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(message)
        logger.info("Email alert sent successfully to %s", ", ".join(recipients))
        return True
    except Exception:
        logger.exception("Failed to send email alert")
        return False


def send_sms_alert(location: str, risk_level: str, probability: float) -> bool:
    """Send a rockfall alert via SMS using Twilio.

    Environment variables used:
    - TWILIO_ACCOUNT_SID: Twilio Account SID (required)
    - TWILIO_AUTH_TOKEN: Twilio Auth Token (required)
    - TWILIO_FROM: Twilio phone number in E.164 format (required)
    - TWILIO_TO: Recipient phone number(s) in E.164, comma-separated (required)
    """

    try:
        from twilio.rest import Client  # type: ignore
    except Exception:
        logger.exception("Twilio SDK not installed or failed to import. Install 'twilio'.")
        return False

    account_sid: Optional[str] = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token: Optional[str] = os.getenv("TWILIO_AUTH_TOKEN")
    from_number: Optional[str] = os.getenv("TWILIO_FROM")
    to_numbers_raw: Optional[str] = os.getenv("TWILIO_TO")

    if not account_sid or not auth_token or not from_number or not to_numbers_raw:
        logger.error(
            "SMS alert not sent: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO are required."
        )
        return False

    to_numbers = [n.strip() for n in to_numbers_raw.split(",") if n.strip()]
    if not to_numbers:
        logger.error("SMS alert not sent: no valid recipients in TWILIO_TO.")
        return False

    probability_text = _format_probability(probability)
    sms_body = (
        "⚠️ Rockfall Risk Alert\n"
        f"Location: {location}\n"
        f"Risk: {risk_level} (Probability: {probability_text})"
    )

    try:
        client = Client(account_sid, auth_token)
        successes = 0
        for recipient in to_numbers:
            try:
                message = client.messages.create(body=sms_body, from_=from_number, to=recipient)
                logger.info("SMS alert sent to %s (sid=%s)", recipient, getattr(message, "sid", "unknown"))
                successes += 1
            except Exception:
                logger.exception("Failed to send SMS to %s", recipient)
        return successes == len(to_numbers)
    except Exception:
        logger.exception("Failed to initialize Twilio client or send SMS alert(s)")
        return False


