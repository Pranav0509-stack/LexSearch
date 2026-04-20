"""
Razorpay Route — auto-split booking fee between partner account and our account
(plan §4.4).

Commission: we keep 20–30% (tunable per partner). Partner's linked account gets
the rest. Razorpay handles settlement + reconciliation.
"""

from dataclasses import dataclass

from app.payments.razorpay_client import client


DEFAULT_COMMISSION_BPS = 2500   # 25.00% — basis points


@dataclass
class RouteSplit:
    our_commission_paise: int
    partner_payout_paise: int


def compute_split(amount_paise: int, commission_bps: int = DEFAULT_COMMISSION_BPS) -> RouteSplit:
    our = (amount_paise * commission_bps) // 10_000
    partner = amount_paise - our
    return RouteSplit(our_commission_paise=our, partner_payout_paise=partner)


def create_route_order(
    *,
    amount_paise: int,
    partner_account_id: str,
    commission_bps: int = DEFAULT_COMMISSION_BPS,
    notes: dict | None = None,
) -> dict:
    """
    Creates an order with a transfer to the partner's linked account. We retain
    `our_commission_paise` on our primary account by default.
    """
    split = compute_split(amount_paise, commission_bps)
    order = client().order.create(
        {
            "amount": amount_paise,
            "currency": "INR",
            "transfers": [
                {
                    "account": partner_account_id,
                    "amount": split.partner_payout_paise,
                    "currency": "INR",
                    "notes": notes or {},
                    "on_hold": False,
                }
            ],
            "notes": notes or {},
        }
    )
    return {"order": order, "split": split.__dict__}
