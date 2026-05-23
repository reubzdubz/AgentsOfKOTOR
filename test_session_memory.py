#!/usr/bin/env python3
"""
Example usage of SessionMemory for KOTOR agent gameplay.
This demonstrates how to initialize, record, and retrieve session data.
"""

import sys
sys.path.insert(0, '/home/reubzdubz/kotoragents')

from shared.session_memory import SessionMemory


def main():
    # Initialize session memory
    session = SessionMemory(session_id="example_session_20260516")
    
    print("=== SESSION MEMORY EXAMPLE ===\n")
    
    # Example 1: Record observations
    print("1. Recording observations...")
    session.record_observation(
        screenshot_class="combat",
        scores={"combat": 0.92, "main_menu": 0.03, "dialogue": 0.05},
        metadata={"enemies_visible": 2, "player_health": 75}
    )
    print("   ✓ Recorded combat state")
    
    # Example 2: Record decisions
    print("\n2. Recording decisions...")
    session.record_decision(
        action="attack_enemy",
        reasoning="Two enemies engaged, player has 75% health.",
        outcome="hit_deal_20_damage"
    )
    print("   ✓ Recorded decision: attack_enemy")
    
    # Example 3: Update objectives
    print("\n3. Updating objectives...")
    session.update_objectives({
        "current": "Escape the Sith Academy",
        "progress": 45,
        "party": [
            {"name": "Player (Revan)", "level": 5, "health": 75},
            {"name": "Bastila", "level": 5, "health": 80}
        ]
    })
    print("   ✓ Updated objectives")
    
    # Example 4: Add dialogue
    print("\n4. Recording dialogue...")
    session.add_conversation("Bastila", "Stay alert. The Sith could be anywhere.")
    session.add_conversation("Player", "I can handle myself.")
    print("   ✓ Logged conversations")
    
    # Example 5: Retrieve recent data
    print("\n5. Retrieving recent context...")
    recent_obs = session.get_recent_observations(limit=3)
    print(f"   Recent observations: {len(recent_obs)}")
    for obs in recent_obs:
        print(f"     - {obs['state']} @ {obs['timestamp']}")
    
    # Example 6: Generate context summary
    print("\n6. Context summary for agent prompts:")
    summary = session.get_context_summary()
    print(summary)
    
    print("\n=== FILES CREATED ===")
    print(f"Session directory: {session.memory_dir}")
    for file in session.memory_dir.glob("*"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
