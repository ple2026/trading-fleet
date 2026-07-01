/**
 * Virtual sub-books. One Alpaca account, four ledger-enforced books — like a real
 * multi-strat: one prime broker, many pods. Every order carries
 * `client_order_id = "<botId>:<signalId>"`, and this ledger tracks per-bot cash,
 * positions, and P&L so no bot can spend another's capital.
 *
 * This is the execution-side counterpart to the Python research layer. It is a
 * phase-2 skeleton: the interfaces and accounting are real; wiring to the Alpaca
 * REST client and the Postgres journal lands with the executor.
 */

export type BotId = 'breakout' | 'arb' | 'catalyst' | 'macro';

export interface SubBook {
  botId: BotId;
  cashUsd: number;
  highWaterUsd: number; // for the per-bot drawdown circuit breaker
  positions: Map<string, Lot>;
}

export interface Lot {
  symbol: string;
  qty: number;
  avgPx: number;
  openedAt: string;
}

const BOT_DD_CIRCUIT_PCT = Number(process.env.BOT_DD_CIRCUIT_PCT ?? '15');

export class Ledger {
  private books = new Map<BotId, SubBook>();

  constructor(seed: Record<BotId, number>) {
    for (const [botId, cash] of Object.entries(seed) as [BotId, number][]) {
      this.books.set(botId, {
        botId,
        cashUsd: cash,
        highWaterUsd: cash,
        positions: new Map(),
      });
    }
  }

  book(botId: BotId): SubBook {
    const b = this.books.get(botId);
    if (!b) throw new Error(`no sub-book for bot ${botId}`);
    return b;
  }

  clientOrderId(botId: BotId, signalId: string): string {
    return `${botId}:${signalId}`;
  }

  /** Equity = cash + marked position value (marks supplied by the caller). */
  equity(botId: BotId, marks: Record<string, number>): number {
    const b = this.book(botId);
    let mkt = 0;
    for (const lot of b.positions.values()) {
      mkt += (marks[lot.symbol] ?? lot.avgPx) * lot.qty;
    }
    return b.cashUsd + mkt;
  }

  /**
   * Per-bot circuit breaker: pause a bot when it draws down more than
   * BOT_DD_CIRCUIT_PCT from its high-water mark. Restart requires human review.
   */
  isTradingHalted(botId: BotId, marks: Record<string, number>): boolean {
    const b = this.book(botId);
    const eq = this.equity(botId, marks);
    b.highWaterUsd = Math.max(b.highWaterUsd, eq);
    const ddPct = ((b.highWaterUsd - eq) / b.highWaterUsd) * 100;
    return ddPct >= BOT_DD_CIRCUIT_PCT;
  }
}
