"""
Enterprise Workflow Scenarios for Compensation Benchmark

This module defines enterprise-scale workflow scenarios that test compensation
at realistic complexity levels for research evaluation.

Scenarios:
1. E-commerce Order Pipeline (inventory → payment → shipping)
2. Cloud Infrastructure Provisioning (VPC → subnet → instance → security)
3. Financial Transaction Processing (auth → transfer → notification)
"""

import json
import time
from typing import Dict, Any, List
from dataclasses import dataclass, field
from langchain_core.tools import tool

from .task_definitions import TaskDefinition, TaskGoal, TaskConstraint, TaskCategory

# State tracking for enterprise workflows
_enterprise_state = {
    "orders": {},
    "inventory": {},
    "payments": {},
    "shipments": {},
    "cloud_resources": {},
    "financial_transactions": {},
}


def reset_enterprise_state():
    """Reset enterprise workflow state"""
    for key in _enterprise_state:
        _enterprise_state[key].clear()


# ============================================================================
# E-commerce Order Pipeline
# ============================================================================

@tool
def reserve_inventory(product_id: str, quantity: int) -> str:
    """Reserve inventory for an order. Fails if insufficient stock."""
    if quantity > 100:  # Simulated stock limit
        return json.dumps({
            "status": "error",
            "message": f"Insufficient inventory for product {product_id}: only 100 available"
        })
    
    reservation_id = f"inv_{product_id}_{int(time.time())}"
    _enterprise_state["inventory"][reservation_id] = {
        "product_id": product_id,
        "quantity": quantity,
        "status": "reserved"
    }
    return json.dumps({
        "reservation_id": reservation_id,
        "product_id": product_id,
        "quantity": quantity,
        "status": "reserved"
    })


@tool
def release_inventory(reservation_id: str) -> str:
    """Release reserved inventory."""
    if reservation_id in _enterprise_state["inventory"]:
        _enterprise_state["inventory"][reservation_id]["status"] = "released"
        return json.dumps({"reservation_id": reservation_id, "status": "released"})
    return json.dumps({"reservation_id": reservation_id, "status": "not_found"})


@tool
def process_payment(order_id: str, amount: float, payment_method: str) -> str:
    """Process payment for an order. Fails if amount > 10000."""
    if amount > 10000:
        return json.dumps({
            "status": "error",
            "message": f"Payment amount {amount} exceeds limit"
        })
    
    payment_id = f"pay_{order_id}_{int(time.time())}"
    _enterprise_state["payments"][payment_id] = {
        "order_id": order_id,
        "amount": amount,
        "payment_method": payment_method,
        "status": "processed"
    }
    return json.dumps({
        "payment_id": payment_id,
        "order_id": order_id,
        "amount": amount,
        "status": "processed"
    })


@tool
def refund_payment(payment_id: str) -> str:
    """Refund a processed payment."""
    if payment_id in _enterprise_state["payments"]:
        _enterprise_state["payments"][payment_id]["status"] = "refunded"
        return json.dumps({"payment_id": payment_id, "status": "refunded"})
    return json.dumps({"payment_id": payment_id, "status": "not_found"})


@tool
def create_shipment(order_id: str, address: str, carrier: str) -> str:
    """Create shipment for an order."""
    shipment_id = f"ship_{order_id}_{int(time.time())}"
    _enterprise_state["shipments"][shipment_id] = {
        "order_id": order_id,
        "address": address,
        "carrier": carrier,
        "status": "created"
    }
    return json.dumps({
        "shipment_id": shipment_id,
        "order_id": order_id,
        "status": "created"
    })


@tool
def cancel_shipment(shipment_id: str) -> str:
    """Cancel a shipment."""
    if shipment_id in _enterprise_state["shipments"]:
        _enterprise_state["shipments"][shipment_id]["status"] = "cancelled"
        return json.dumps({"shipment_id": shipment_id, "status": "cancelled"})
    return json.dumps({"shipment_id": shipment_id, "status": "not_found"})


# ============================================================================
# Cloud Infrastructure Provisioning
# ============================================================================

@tool
def create_vpc(vpc_name: str, cidr_block: str) -> str:
    """Create a VPC. Fails if VPC limit exceeded."""
    if len([v for v in _enterprise_state["cloud_resources"].values() if v.get("type") == "vpc"]) >= 5:
        return json.dumps({
            "status": "error",
            "message": "VPC limit exceeded: maximum 5 VPCs allowed"
        })
    
    vpc_id = f"vpc_{vpc_name}_{int(time.time())}"
    _enterprise_state["cloud_resources"][vpc_id] = {
        "type": "vpc",
        "name": vpc_name,
        "cidr_block": cidr_block,
        "status": "created"
    }
    return json.dumps({
        "vpc_id": vpc_id,
        "name": vpc_name,
        "status": "created"
    })


@tool
def delete_vpc(vpc_id: str) -> str:
    """Delete a VPC."""
    if vpc_id in _enterprise_state["cloud_resources"]:
        _enterprise_state["cloud_resources"][vpc_id]["status"] = "deleted"
        return json.dumps({"vpc_id": vpc_id, "status": "deleted"})
    return json.dumps({"vpc_id": vpc_id, "status": "not_found"})


@tool
def create_subnet(vpc_id: str, subnet_name: str, cidr_block: str) -> str:
    """Create a subnet in a VPC."""
    subnet_id = f"subnet_{subnet_name}_{int(time.time())}"
    _enterprise_state["cloud_resources"][subnet_id] = {
        "type": "subnet",
        "vpc_id": vpc_id,
        "name": subnet_name,
        "cidr_block": cidr_block,
        "status": "created"
    }
    return json.dumps({
        "subnet_id": subnet_id,
        "vpc_id": vpc_id,
        "status": "created"
    })


@tool
def delete_subnet(subnet_id: str) -> str:
    """Delete a subnet."""
    if subnet_id in _enterprise_state["cloud_resources"]:
        _enterprise_state["cloud_resources"][subnet_id]["status"] = "deleted"
        return json.dumps({"subnet_id": subnet_id, "status": "deleted"})
    return json.dumps({"subnet_id": subnet_id, "status": "not_found"})


@tool
def launch_instance(subnet_id: str, instance_type: str, image_id: str) -> str:
    """Launch a compute instance. Fails if instance limit exceeded."""
    existing_instances = [r for r in _enterprise_state["cloud_resources"].values() if r.get("type") == "instance"]
    if len(existing_instances) >= 10:
        return json.dumps({
            "status": "error",
            "message": "Instance limit exceeded: maximum 10 instances allowed"
        })
    
    instance_id = f"i_{subnet_id}_{int(time.time())}"
    _enterprise_state["cloud_resources"][instance_id] = {
        "type": "instance",
        "subnet_id": subnet_id,
        "instance_type": instance_type,
        "image_id": image_id,
        "status": "running"
    }
    return json.dumps({
        "instance_id": instance_id,
        "subnet_id": subnet_id,
        "status": "running"
    })


@tool
def terminate_instance(instance_id: str) -> str:
    """Terminate a compute instance."""
    if instance_id in _enterprise_state["cloud_resources"]:
        _enterprise_state["cloud_resources"][instance_id]["status"] = "terminated"
        return json.dumps({"instance_id": instance_id, "status": "terminated"})
    return json.dumps({"instance_id": instance_id, "status": "not_found"})


@tool
def create_security_group(vpc_id: str, group_name: str, rules: List[Dict[str, Any]]) -> str:
    """Create a security group."""
    sg_id = f"sg_{group_name}_{int(time.time())}"
    _enterprise_state["cloud_resources"][sg_id] = {
        "type": "security_group",
        "vpc_id": vpc_id,
        "name": group_name,
        "rules": rules,
        "status": "created"
    }
    return json.dumps({
        "security_group_id": sg_id,
        "vpc_id": vpc_id,
        "status": "created"
    })


@tool
def delete_security_group(security_group_id: str) -> str:
    """Delete a security group."""
    if security_group_id in _enterprise_state["cloud_resources"]:
        _enterprise_state["cloud_resources"][security_group_id]["status"] = "deleted"
        return json.dumps({"security_group_id": security_group_id, "status": "deleted"})
    return json.dumps({"security_group_id": security_group_id, "status": "not_found"})


# ============================================================================
# Financial Transaction Processing
# ============================================================================

@tool
def authenticate_user(user_id: str, credentials: str) -> str:
    """Authenticate a user. Fails if invalid credentials."""
    if credentials == "invalid":
        return json.dumps({
            "status": "error",
            "message": "Authentication failed: invalid credentials"
        })
    
    auth_token = f"auth_{user_id}_{int(time.time())}"
    _enterprise_state["financial_transactions"][auth_token] = {
        "type": "authentication",
        "user_id": user_id,
        "status": "authenticated"
    }
    return json.dumps({
        "auth_token": auth_token,
        "user_id": user_id,
        "status": "authenticated"
    })


@tool
def revoke_authentication(auth_token: str) -> str:
    """Revoke an authentication token."""
    if auth_token in _enterprise_state["financial_transactions"]:
        _enterprise_state["financial_transactions"][auth_token]["status"] = "revoked"
        return json.dumps({"auth_token": auth_token, "status": "revoked"})
    return json.dumps({"auth_token": auth_token, "status": "not_found"})


@tool
def transfer_funds(from_account: str, to_account: str, amount: float, auth_token: str) -> str:
    """Transfer funds between accounts. Fails if amount > 50000 or insufficient balance."""
    if amount > 50000:
        return json.dumps({
            "status": "error",
            "message": f"Transfer amount {amount} exceeds limit"
        })
    
    if auth_token not in _enterprise_state["financial_transactions"]:
        return json.dumps({
            "status": "error",
            "message": "Authentication required"
        })
    
    transaction_id = f"txn_{from_account}_{int(time.time())}"
    _enterprise_state["financial_transactions"][transaction_id] = {
        "type": "transfer",
        "from_account": from_account,
        "to_account": to_account,
        "amount": amount,
        "auth_token": auth_token,
        "status": "completed"
    }
    return json.dumps({
        "transaction_id": transaction_id,
        "from_account": from_account,
        "to_account": to_account,
        "amount": amount,
        "status": "completed"
    })


@tool
def reverse_transfer(transaction_id: str) -> str:
    """Reverse a funds transfer."""
    if transaction_id in _enterprise_state["financial_transactions"]:
        _enterprise_state["financial_transactions"][transaction_id]["status"] = "reversed"
        return json.dumps({"transaction_id": transaction_id, "status": "reversed"})
    return json.dumps({"transaction_id": transaction_id, "status": "not_found"})


@tool
def send_notification(user_id: str, message: str, channel: str) -> str:
    """Send a notification to a user."""
    notification_id = f"notif_{user_id}_{int(time.time())}"
    _enterprise_state["financial_transactions"][notification_id] = {
        "type": "notification",
        "user_id": user_id,
        "message": message,
        "channel": channel,
        "status": "sent"
    }
    return json.dumps({
        "notification_id": notification_id,
        "user_id": user_id,
        "status": "sent"
    })


# ============================================================================
# Compensation Mappings
# ============================================================================

ENTERPRISE_COMPENSATION_MAPPING = {
    # E-commerce
    "reserve_inventory": "release_inventory",
    "process_payment": "refund_payment",
    "create_shipment": "cancel_shipment",
    # Cloud
    "create_vpc": "delete_vpc",
    "create_subnet": "delete_subnet",
    "launch_instance": "terminate_instance",
    "create_security_group": "delete_security_group",
    # Financial
    "authenticate_user": "revoke_authentication",
    "transfer_funds": "reverse_transfer",
}


# ============================================================================
# Enterprise Task Definitions
# ============================================================================

ENTERPRISE_TASKS = {
    "E1": TaskDefinition(
        task_id="E1",
        name="E-commerce Order Pipeline",
        category=TaskCategory.LOGISTICS,
        description="Complete order processing: reserve inventory → process payment → create shipment",
        goals=[
            TaskGoal("reserve_inventory", "Reserve inventory for order", 0.33),
            TaskGoal("process_payment", "Process payment successfully", 0.33),
            TaskGoal("create_shipment", "Create shipment for order", 0.34),
        ],
        constraints=[
            TaskConstraint("inventory_availability", "resource", "Inventory must be available", {}),
            TaskConstraint("payment_limit", "resource", "Payment amount must be within limit", {}),
            TaskConstraint("order_deadline", "deadline", "Order must complete within deadline", {}),
        ],
        resources={
            "products": ["product_1", "product_2"],
            "payment_methods": ["credit_card", "paypal"],
            "carriers": ["ups", "fedex"]
        }
    ),
    
    "E2": TaskDefinition(
        task_id="E2",
        name="Cloud Infrastructure Provisioning",
        category=TaskCategory.SUPPLY_CHAIN,
        description="Provision cloud infrastructure: VPC → subnet → instance → security group",
        goals=[
            TaskGoal("create_vpc", "Create VPC", 0.25),
            TaskGoal("create_subnet", "Create subnet in VPC", 0.25),
            TaskGoal("launch_instance", "Launch compute instance", 0.25),
            TaskGoal("create_security_group", "Create security group", 0.25),
        ],
        constraints=[
            TaskConstraint("vpc_limit", "capacity", "Maximum 5 VPCs", {}),
            TaskConstraint("instance_limit", "capacity", "Maximum 10 instances", {}),
            TaskConstraint("dependency_order", "dependency", "Must create VPC before subnet, subnet before instance", {}),
        ],
        resources={
            "vpc_names": ["production", "staging"],
            "instance_types": ["t2.micro", "t2.small"],
            "image_ids": ["ami-12345", "ami-67890"]
        }
    ),
    
    "E3": TaskDefinition(
        task_id="E3",
        name="Financial Transaction Processing",
        category=TaskCategory.LOGISTICS,
        description="Process financial transaction: authenticate → transfer funds → send notification",
        goals=[
            TaskGoal("authenticate_user", "Authenticate user", 0.33),
            TaskGoal("transfer_funds", "Transfer funds between accounts", 0.33),
            TaskGoal("send_notification", "Send transaction notification", 0.34),
        ],
        constraints=[
            TaskConstraint("authentication_required", "dependency", "Must authenticate before transfer", {}),
            TaskConstraint("transfer_limit", "resource", "Transfer amount must be within limit", {}),
            TaskConstraint("transaction_deadline", "deadline", "Transaction must complete within deadline", {}),
        ],
        resources={
            "users": ["user_1", "user_2"],
            "accounts": ["account_1", "account_2"],
            "channels": ["email", "sms"]
        }
    ),
}
