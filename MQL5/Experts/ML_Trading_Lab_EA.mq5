//+------------------------------------------------------------------+
//|                                              BB_EMA_V3_State.mq5 |
//|                                      Generated from Pine Script  |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property link      ""
#property version   "1.00"

#include <Trade\Trade.mqh>

//--- Inputs
input group "=== Trade Settings ==="
input double   InpLotSize       = 0.1;       // Lot Size
input ulong    InpMagicNumber   = 123456;    // Magic Number

input group "=== Bollinger Bands (Red - Inner) ==="
input int      InpBBLen         = 20;        // BB Length
input double   InpBBMult        = 1.0;       // BB StdDev

input group "=== EMA Bollinger Bands (Green - Outer) ==="
input int      InpEMALen        = 20;        // EMA Length
input int      InpSmoothLen     = 20;        // Smoothing Length
input double   InpSmoothMult    = 2.5;       // EMA BB StdDev

input group "=== Squeeze & Slope ==="
input double   InpSqzThreshold  = 5.0;       // Squeeze Gap Threshold (Points) - Adjust for your broker's digits
input int      InpSlopeLen      = 3;         // Slope Lookback Bars

input group "=== 3M Filters ==="
input bool     InpUse3MEntry    = true;      // Require 3M Alignment for Entry
input bool     InpUse3MExit     = true;      // Require 3M Misalignment for Exit

//--- Global Variables
CTrade trade;
int emaM1_Handle;
int emaM3_Handle;

//--- State Machine Enum
enum ENUM_EA_STATE {
   STATE_NO_POS,  // Flat
   STATE_LONG,    // Currently in a Long trade
   STATE_SHORT    // Currently in a Short trade
};
ENUM_EA_STATE eaState = STATE_NO_POS;

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
int OnInit() {
   trade.SetExpertMagicNumber(InpMagicNumber);
   
   // Create EMA handles for M1 (Current) and M3
   emaM1_Handle = iMA(_Symbol, PERIOD_CURRENT, InpEMALen, 0, MODE_EMA, PRICE_CLOSE);
   emaM3_Handle = iMA(_Symbol, PERIOD_M3, InpEMALen, 0, MODE_EMA, PRICE_CLOSE);
   
   if(emaM1_Handle == INVALID_HANDLE || emaM3_Handle == INVALID_HANDLE) {
      Print("Error creating indicator handles!");
      return(INIT_FAILED);
   }
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Deinitialization                                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   if(emaM1_Handle != INVALID_HANDLE) IndicatorRelease(emaM1_Handle);
   if(emaM3_Handle != INVALID_HANDLE) IndicatorRelease(emaM3_Handle);
}

//+------------------------------------------------------------------+
//| Math Helpers                                                     |
//+------------------------------------------------------------------+
double CalcSMA(const double &arr[], int period, int shift) {
   double sum = 0;
   for(int i = shift; i < shift + period; i++) sum += arr[i];
   return (period > 0) ? (sum / period) : 0;
}

double CalcStdDev(const double &arr[], int period, double mean, int shift) {
   double sum_sq = 0;
   for(int i = shift; i < shift + period; i++) {
      double diff = arr[i] - mean;
      sum_sq += diff * diff;
   }
   return (period > 0) ? MathSqrt(sum_sq / period) : 0;
}

//+------------------------------------------------------------------+
//| New Bar Checker                                                  |
//+------------------------------------------------------------------+
bool IsNewBar() {
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBarTime != lastBarTime) {
      lastBarTime = currentBarTime;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Sync EA State with actual MT5 positions (e.g. if manual close)   |
//+------------------------------------------------------------------+
void SyncState() {
   bool hasLong = false, hasShort = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber) {
         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) hasLong = true;
         if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) hasShort = true;
      }
   }
   if(eaState == STATE_LONG && !hasLong) eaState = STATE_NO_POS;
   if(eaState == STATE_SHORT && !hasShort) eaState = STATE_NO_POS;
   if(eaState == STATE_NO_POS && hasLong) eaState = STATE_LONG;
   if(eaState == STATE_NO_POS && hasShort) eaState = STATE_SHORT;
}

//+------------------------------------------------------------------+
//| Close Specific Position Type                                     |
//+------------------------------------------------------------------+
void ClosePositions(ENUM_POSITION_TYPE posType) {
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber && PositionGetInteger(POSITION_TYPE) == posType) {
         trade.PositionClose(ticket);
      }
   }
}

//+------------------------------------------------------------------+
//| Main Tick Function                                               |
//+------------------------------------------------------------------+
void OnTick() {
   if(!IsNewBar()) return;
   SyncState(); // Ensure EA state matches reality
   
   // ---------------------------------------------------------
   // 1. GET M1 DATA
   // ---------------------------------------------------------
   int reqBars = InpSmoothLen + InpSlopeLen + 5;
   double closeM1[], emaM1[];
   ArraySetAsSeries(closeM1, true);
   ArraySetAsSeries(emaM1, true);
   
   if(CopyClose(_Symbol, PERIOD_CURRENT, 0, reqBars, closeM1) <= 0) return;
   if(CopyBuffer(emaM1_Handle, 0, 0, reqBars, emaM1) <= 0) return;

   // ---------------------------------------------------------
   // 2. CALCULATE M1 BANDS (Current [0] and Previous [1])
   // ---------------------------------------------------------
   double bbUp[2], bbDn[2], emaUp[2], emaDn[2], totalGap[2];
   
   for(int i = 0; i <= 1; i++) {
      double bbBasis = CalcSMA(closeM1, InpBBLen, i);
      double bbDev = InpBBMult * CalcStdDev(closeM1, InpBBLen, bbBasis, i);
      bbUp[i] = bbBasis + bbDev;
      bbDn[i] = bbBasis - bbDev;
      
      double smBasis = CalcSMA(emaM1, InpSmoothLen, i);
      double smDev = InpSmoothMult * CalcStdDev(emaM1, InpSmoothLen, smBasis, i);
      emaUp[i] = smBasis + smDev;
      emaDn[i] = smBasis - smDev;
      
      totalGap[i] = (emaUp[i] - bbUp[i]) + (emaDn[i] - bbDn[i]);
   }

   // ---------------------------------------------------------
   // 3. CALCULATE M1 SLOPE LOOKBACK VALUES
   // ---------------------------------------------------------
   int sl = InpSlopeLen;
   
   double bbUp_sl   = CalcSMA(closeM1, InpBBLen, sl) + InpBBMult * CalcStdDev(closeM1, InpBBLen, CalcSMA(closeM1, InpBBLen, sl), sl);
   double bbDn_sl   = CalcSMA(closeM1, InpBBLen, sl) - InpBBMult * CalcStdDev(closeM1, InpBBLen, CalcSMA(closeM1, InpBBLen, sl), sl);
   
   double emaUp_sl  = CalcSMA(emaM1, InpSmoothLen, sl) + InpSmoothMult * CalcStdDev(emaM1, InpSmoothLen, CalcSMA(emaM1, InpSmoothLen, sl), sl);
   double emaDn_sl  = CalcSMA(emaM1, InpSmoothLen, sl) - InpSmoothMult * CalcStdDev(emaM1, InpSmoothLen, CalcSMA(emaM1, InpSmoothLen, sl), sl);
   
   double emaUp_1_sl = CalcSMA(emaM1, InpSmoothLen, 1+sl) + InpSmoothMult * CalcStdDev(emaM1, InpSmoothLen, CalcSMA(emaM1, InpSmoothLen, 1+sl), 1+sl);
   double emaDn_1_sl = CalcSMA(emaM1, InpSmoothLen, 1+sl) - InpSmoothMult * CalcStdDev(emaM1, InpSmoothLen, CalcSMA(emaM1, InpSmoothLen, 1+sl), 1+sl);

   // ---------------------------------------------------------
   // 4. M1 LOGIC CONDITIONS
   // ---------------------------------------------------------
   bool isSqueezed  = totalGap[1] <= InpSqzThreshold; 
   bool isExpanding = totalGap[0] > totalGap[1];
   
   bool bbUpRising  = bbUp[0] > bbUp_sl;
   bool bbDnFalling = bbDn[0] < bbDn_sl;
   
   // LONG: BB lower crosses above EMA BB lower + BB upper rising + expanding
   bool longCond1M = (bbDn[1] <= emaDn[1] && bbDn[0] > emaDn[0]) && bbUpRising && isExpanding;
   
   // SHORT: BB upper crosses below EMA BB upper + BB lower falling + expanding
   bool shortCond1M = (bbUp[1] >= emaUp[1] && bbUp[0] < emaUp[0]) && bbDnFalling && isExpanding;

   // ---------------------------------------------------------
   // 5. M3 LOGIC (Strictly using CLOSED bar index 1)
   // ---------------------------------------------------------
   bool m3_longAlign = false;
   bool m3_shortAlign = false;
   
   if(InpUse3MEntry || InpUse3MExit) {
      double closeM3[], emaM3[];
      ArraySetAsSeries(closeM3, true);
      ArraySetAsSeries(emaM3, true);
      
      if(CopyClose(_Symbol, PERIOD_M3, 0, reqBars, closeM3) > 0 && CopyBuffer(emaM3_Handle, 0, 0, reqBars, emaM3) > 0) {
         int m = 1; // Index 1 = Last CLOSED 3M bar (prevents flickering)
         
         double m3_bbBasis = CalcSMA(closeM3, InpBBLen, m);
         double m3_bbDev = InpBBMult * CalcStdDev(closeM3, InpBBLen, m3_bbBasis, m);
         double m3_bbUp = m3_bbBasis + m3_bbDev;
         double m3_bbDn = m3_bbBasis - m3_bbDev;
         
         double m3_smBasis = CalcSMA(emaM3, InpSmoothLen, m);
         double m3_smDev = InpSmoothMult * CalcStdDev(emaM3, InpSmoothLen, m3_smBasis, m);
         double m3_emaUp = m3_smBasis + m3_smDev;
         double m3_emaDn = m3_smBasis - m3_smDev;
         
         m3_longAlign  = (m3_bbUp > m3_emaUp && m3_bbDn > m3_emaDn);
         m3_shortAlign = (m3_bbUp < m3_emaUp && m3_bbDn < m3_emaDn);
      }
   }

   // ---------------------------------------------------------
   // 6. EXIT CONDITIONS
   // ---------------------------------------------------------
   bool emaUpRisingPrev  = emaUp[1] > emaUp_1_sl;
   bool emaUpFallingCurr = emaUp[0] < emaUp_sl;
   bool longExitSlope    = emaUpRisingPrev && emaUpFallingCurr;
   bool longExit         = longExitSlope && (InpUse3MExit ? !m3_longAlign : true);
   
   bool emaDnFallingPrev = emaDn[1] < emaDn_1_sl;
   bool emaDnRisingCurr  = emaDn[0] > emaDn_sl;
   bool shortExitSlope   = emaDnFallingPrev && emaDnRisingCurr;
   bool shortExit        = shortExitSlope && (InpUse3MExit ? !m3_shortAlign : true);

   // ---------------------------------------------------------
   // 7. STATE MACHINE EXECUTION (Fixes "Many Signals" issue)
   // ---------------------------------------------------------
   if(eaState == STATE_NO_POS) {
      bool finalLong  = longCond1M  && (InpUse3MEntry ? m3_longAlign : true);
      bool finalShort = shortCond1M && (InpUse3MEntry ? m3_shortAlign : true);
      
      if(finalLong) {
         if(trade.Buy(InpLotSize, _Symbol)) {
            eaState = STATE_LONG;
            Print("EA State -> LONG");
         }
      } 
      else if(finalShort) {
         if(trade.Sell(InpLotSize, _Symbol)) {
            eaState = STATE_SHORT;
            Print("EA State -> SHORT");
         }
      }
   } 
   else if(eaState == STATE_LONG) {
      if(longExit) {
         ClosePositions(POSITION_TYPE_BUY);
         eaState = STATE_NO_POS;
         Print("EA State -> FLAT (Long Exit)");
      }
   } 
   else if(eaState == STATE_SHORT) {
      if(shortExit) {
         ClosePositions(POSITION_TYPE_SELL);
         eaState = STATE_NO_POS;
         Print("EA State -> FLAT (Short Exit)");
      }
   }
}
//+------------------------------------------------------------------+