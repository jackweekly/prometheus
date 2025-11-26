import random
import argparse

class VendingMachine:
    """
    Represents the state of a single vending machine.
    """
    def __init__(self, initial_inventory=50, initial_price=1.5, cost_per_item=0.5):
        self.inventory = initial_inventory
        self.price = initial_price
        self.cost_per_item = cost_per_item
        self.revenue = 0.0
        self.profit = 0.0
        self.stock_costs = 0.0

    def restock(self, quantity):
        """Adds items to inventory and accounts for the cost."""
        self.inventory += quantity
        self.stock_costs += quantity * self.cost_per_item
        print(f"ACTION: Restocked {quantity} items. New inventory: {self.inventory}.")

    def update_finances(self, sales):
        """Updates revenue and profit."""
        daily_revenue = sales * self.price
        self.revenue += daily_revenue
        self.profit = self.revenue - self.stock_costs

    def __str__(self):
        return f"Inv: {self.inventory}, Price: ${self.price:.2f}, Revenue: ${self.revenue:.2f}, Profit: ${self.profit:.2f}"

class VendingSimulation:
    """
    Encapsulates the Vending-Bench simulation environment.
    """
    def __init__(self, agent_type="llm"):
        self.machine = VendingMachine()
        self.agent_type = agent_type
        self.day = 0

    def simulate_day(self, base_demand=20, price_sensitivity=10):
        """
        Simulates a single day of sales with a slightly more realistic demand model.
        """
        # Demand decreases as price increases
        demand = int(base_demand - (self.machine.price * price_sensitivity))
        demand = max(0, demand) # Ensure demand is not negative
        
        sales = min(self.machine.inventory, demand)
        
        self.machine.inventory -= sales
        self.machine.update_finances(sales)
        
        return sales

    def get_state_prompt(self):
        """
        Generates a string prompt describing the current state for the LLM agent.
        """
        prompt = f"""
        Vending Machine Status:
        - Current Day: {self.day}
        - Inventory: {self.machine.inventory} items
        - Current Price: ${self.machine.price:.2f}
        - Total Revenue: ${self.machine.revenue:.2f}
        - Total Profit: ${self.machine.profit:.2f}

        Your goal is to maximize profit over 30 days.
        Your inventory is running low. A restock costs ${self.machine.cost_per_item:.2f} per item.
        High prices might reduce sales. Low prices might hurt profit margins.
        
        Choose one of the following actions:
        1. {"action": "increase_price", "value": 0.25}
        2. {"action": "decrease_price", "value": 0.25}
        3. {"action": "restock", "value": 50}
        4. {"action": "do_nothing"}
        """
        return prompt

    def llm_agent_action(self):
        """
        A placeholder for a real LLM agent.
        This function will take the state prompt and return a structured action.
        For now, it uses simple rules to mimic basic reasoning.
        """
        prompt = self.get_state_prompt()
        print(f"\n[Agent Prompt]\n{prompt}")

        # Rule-based logic to simulate an LLM
        if self.machine.inventory < 10:
            # If inventory is low, always restock
            action = {"action": "restock", "value": 50}
        elif self.machine.price > 2.0:
            # If price is high, consider lowering it
            action = {"action": "decrease_price", "value": 0.25}
        elif self.machine.price < 1.0:
            # If price is low, consider increasing it
            action = {"action": "increase_price", "value": 0.25}
        else:
            # Otherwise, do nothing
            action = {"action": "do_nothing"}
        
        print(f"LLM Agent decision: {action}")
        return action

    def run(self, num_days=30):
        """Main simulation loop."""
        print("🚀 Initializing Vending-Bench Simulation...")
        print(f"Agent Type: {self.agent_type}")
        print(f"Initial state: {self.machine}")

        for i in range(1, num_days + 1):
            self.day = i
            print(f"\n--- Day {self.day} ---")

            # Agent takes an action
            if self.agent_type == "llm":
                agent_decision = self.llm_agent_action()
                action_type = agent_decision.get("action")
                value = agent_decision.get("value")

                if action_type == "increase_price":
                    self.machine.price += value
                elif action_type == "decrease_price":
                    self.machine.price = max(self.machine.cost_per_item, self.machine.price - value)
                elif action_type == "restock":
                    self.machine.restock(value)

            # Simulate sales for the day
            sales = self.simulate_day()
            print(f"Sales for the day: {sales}")
            print(f"End of day state: {self.machine}")

        self.generate_report()

    def generate_report(self):
        """Prints a summary of the simulation results."""
        print("\n--- 📈 Simulation Complete ---")
        print(f"Total Days: {self.day}")
        print(f"Final Inventory: {self.machine.inventory}")
        print(f"Final Price: ${self.machine.price:.2f}")
        print(f"Total Revenue: ${self.machine.revenue:.2f}")
        print(f"Total Stock Costs: ${self.machine.stock_costs:.2f}")
        print(f"Survival Duration: {self.day} days")
        print("---------------------------------")
        print(f"🏆 Total Profit: ${self.machine.profit:.2f}")
        print("---------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Run the Vending-Bench Simulation")
    parser.add_argument("--agent", type=str, default="llm", choices=["llm", "random"],
                        help="The type of agent to use in the simulation.")
    args = parser.parse_args()
    
    sim = VendingSimulation(agent_type=args.agent)
    sim.run()

if __name__ == "__main__":
    main()
