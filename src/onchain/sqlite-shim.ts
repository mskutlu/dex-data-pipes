import BetterSqlite3, { Database as BetterDB, Statement as BetterStmt } from 'better-sqlite3';

// Minimal runtime-compatible shim to satisfy dex-data-pipes imports of `node:sqlite`.
// It wraps better-sqlite3 to provide DatabaseSync and StatementSync with the methods used there.

export class StatementSync {
  private readonly stmt: BetterStmt;
  constructor(stmt: BetterStmt) {
    this.stmt = stmt;
  }
  run(params?: any): any {
    // better-sqlite3 accepts either single object or varargs
    if (Array.isArray(params)) return (this.stmt as any).run(...params);
    if (params === undefined) return this.stmt.run();
    
    // Handle named parameters with potential missing values
    if (typeof params === 'object' && params !== null) {
      // Extract parameter names from SQL query
      const sql = this.stmt.source;
      const namedParams = sql.match(/:\w+/g) || [];
      const paramNames = namedParams.map(p => p.slice(1)); // Remove : prefix
      
      // Ensure all named parameters are present, set to null if missing
      const completeParams = { ...params };
      for (const name of paramNames) {
        if (!(name in completeParams)) {
          completeParams[name] = null;
        }
      }
      return this.stmt.run(completeParams);
    }
    
    return this.stmt.run(params);
  }
  all(...params: any[]): any[] {
    return (this.stmt as any).all(...params);
  }
  get(...params: any[]): any {
    return (this.stmt as any).get(...params);
  }
}

export class DatabaseSync {
  private readonly db: BetterDB;
  constructor(path: string) {
    this.db = new BetterSqlite3(path);
  }
  exec(sql: string): void {
    this.db.exec(sql);
  }
  prepare(sql: string): StatementSync {
    return new StatementSync(this.db.prepare(sql));
  }
  transaction<T extends (...args: any[]) => any>(fn: T): T {
    return this.db.transaction(fn) as any;
  }
  // Expose raw database if ever needed
  get raw(): BetterDB {
    return this.db;
  }
}
