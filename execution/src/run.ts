/**
 * Execution entrypoint (phase 2). The Python research layer emits sized signals
 * (via the journal / a signals table); this process reads them, enforces the
 * per-bot sub-book ledger + circuit breakers, and submits bracket orders to
 * Alpaca paper. Kept as a documented skeleton until the research layer clears its
 * promotion gates — see docs/FLEET_PLAN.md §10.
 */

import { Ledger, type BotId } from './ledger.js';

const SEED: Record<BotId, number> = {
  breakout: 10_000,
  arb: 10_000,
  catalyst: 10_000,
  macro: 10_000,
};

function main(): void {
  const ledger = new Ledger(SEED);
  // Phase 2 wiring lands here:
  //   1. pull today's sized signals for each bot from the journal
  //   2. for each: skip if ledger.isTradingHalted(botId, marks)
  //   3. submit bracket order (entry + stop + target) to Alpaca paper
  //   4. reconcile fills back into the ledger and the journal (slippage_bps)
  for (const botId of Object.keys(SEED) as BotId[]) {
    const book = ledger.book(botId);
    console.log(`${botId}: $${book.cashUsd.toLocaleString()} cash, ${book.positions.size} positions`);
  }
  console.log('\nExecutor skeleton — see docs/FLEET_PLAN.md phase 6 for go-live.');
}

main();
