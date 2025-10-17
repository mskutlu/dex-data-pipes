import path from 'path';
import { ClickhouseState } from '@sqd-pipes/core';
import { createClickhouseClient, ensureTables, toUnixTime } from '../../clickhouse';
import { chRetry, createLogger } from '../../utils';
import { getConfig } from './config';
import { SolanaSwapsStream } from '../../../streams/solana/swaps-stream';
import { PriceExtendStream } from '../../../streams/solana/price-extend-stream';
import { asDecimalString, timeIt } from '../../../streams/solana/utils';

const config = getConfig();

const logger = createLogger('solana dex swaps');

logger.info(`Local database: ${config.dbPath}`);
logger.info(`Cache dump path: ${config.cacheDumpPath}`);

async function main() {
  // Serialize BigInts as string
  BigInt.prototype['toJSON'] = function () {
    return this.toString();
  };
  Error.stackTraceLimit = 1000;
  const clickhouse = await createClickhouseClient({
    capture_enhanced_stack_trace: true,
    clickhouse_settings: {
      http_receive_timeout: 900,
      receive_timeout: 900,
    },
    request_timeout: 900_000,
    // log: {
    //   level: ClickHouseLogLevel.TRACE,
    // },
  });
  await ensureTables(
    clickhouse,
    path.join(__dirname, 'sql'),
    undefined,
    process.env.CLICKHOUSE_DB || 'default',
  );

  const ds = new SolanaSwapsStream({
    portal: config.portalUrl,
    blockRange: {
      from: config.blockFrom,
      to: config.blockTo,
    },
    args: {
      dbPath: config.dbPath,
      onlyMeta: config.onlyMeta,
    },
    logger,
    state: new ClickhouseState(clickhouse, {
      database: process.env.CLICKHOUSE_DB,
      table: `sync_status`,
      id: `solana_swaps`,
      onRollback: async ({ state, latest }) => {
        if (!latest.timestamp) {
          return; // fresh table
        }
        await state.removeAllRows({
          table: `solana_swaps_raw`,
          where: `timestamp > ${latest.timestamp}`,
        });
        // TODO: What about tokens metadata?
      },
    }),
  });
  ds.initialize();

  const stream = await ds.stream();
  for await (const swaps of stream.pipeThrough(
    await new PriceExtendStream({
      clickhouse,
      cacheDumpPath: config.cacheDumpPath,
      cacheDumpIntervalBlocks: config.cacheDumpIntervalBlocks,
    }).pipe(),
  )) {
    await timeIt(
      logger,
      `Inserting swaps to Clickhouse`,
      async () => {
        await chRetry(
          () =>
            clickhouse.insert({
              table: `solana_swaps_raw`,
              values: swaps.map((s) => {
                const obj = {
                  // Name of the DEX
                  dex: s.type,
                  // Blockchain data
                  block_number: s.block.number,
                  transaction_hash: s.transaction.hash,
                  transaction_index: s.transaction.index,
                  instruction_address: s.instruction.address,
                  // Account which executed the swap
                  account: s.account,
                  // Mint accounts of the tokens
                  token_a: s.baseToken.mintAcc,
                  token_b: s.quoteToken.mintAcc,
                  // Amounts of the tokens exchanged
                  amount_a: asDecimalString(s.baseToken.amount, s.baseToken.decimals),
                  amount_b: asDecimalString(s.quoteToken.amount, s.quoteToken.decimals),
                  // Tokens metadata
                  token_a_creation_date: s.baseToken.createdAt
                    ? toUnixTime(new Date(s.baseToken.createdAt))
                    : 0,
                  token_b_creation_date: s.quoteToken.createdAt
                    ? toUnixTime(new Date(s.quoteToken.createdAt))
                    : 0,
                  token_a_decimals: s.baseToken.decimals,
                  token_a_symbol: s.baseToken.symbol || '[unknown]',
                  token_b_decimals: s.quoteToken.decimals,
                  token_b_symbol: s.quoteToken.symbol || '[unknown]',
                  // Token prices
                  token_a_usdc_price: s.baseToken.priceUsdc,
                  token_b_usdc_price: s.quoteToken.priceUsdc,
                  token_a_pricing_pool: s.baseToken.usdcPricingPool?.address || '',
                  token_b_pricing_pool: s.quoteToken.usdcPricingPool?.address || '',
                  // Token issuances
                  token_a_issuance: s.baseToken.issuance?.toString() || 0,
                  token_b_issuance: s.quoteToken.issuance?.toString() || 0,
                  // Trader stats
                  token_a_balance: s.baseToken.balance,
                  token_b_balance: s.quoteToken.balance,
                  token_a_profit_usdc: s.baseToken.positionExitSummary?.profitUsdc || 0,
                  token_b_profit_usdc: s.quoteToken.positionExitSummary?.profitUsdc || 0,
                  token_a_cost_usdc: s.baseToken.positionExitSummary?.entryCostUsdc || 0,
                  token_b_cost_usdc: s.quoteToken.positionExitSummary?.entryCostUsdc || 0,
                  token_a_wins: s.baseToken.wins,
                  token_b_wins: s.quoteToken.wins,
                  token_a_loses: s.baseToken.loses,
                  token_b_loses: s.quoteToken.loses,
                  // Timestamp
                  timestamp: toUnixTime(s.timestamp),
                  // Slippage
                  slippage_pct: s.slippagePct,
                  // Pool data
                  pool_address: s.poolAddress,
                  pool_token_a_reserve: asDecimalString(
                    s.baseToken.reserves || 0n,
                    s.baseToken.decimals,
                  ),
                  pool_token_b_reserve: asDecimalString(
                    s.quoteToken.reserves || 0n,
                    s.quoteToken.decimals,
                  ),
                  sign: 1,
                };
                return obj;
              }),
              format: 'JSONEachRow',
            }),
          { logger, desc: `Insert ${swaps.length} swaps to Clickhouse` },
        );
        await ds.ack();
      },
      {
        numSwaps: swaps.length,
      },
    );
  }
}

void main();
