//+------------------------------------------------------------------+
//|                                     NAS100_MicroStructure_LSTM   |
//|                                              Automated AI Trader |
//+------------------------------------------------------------------+
#property copyright "Automated AI Trader"
#property link      ""
#property version   "1.00"

// Embed ONNX model as resource
#resource "nas100_lstm.onnx" as uchar ExtModelData[]

#include <Trade\Trade.mqh>
#include <Math\Stat\Math.mqh>

input double InpRiskPercent = 1.0; // Risk per trade (%)

// Globals
CTrade trade;
long model_handle = INVALID_HANDLE;
double min_val[5] = {-0.08641843593299904, -0.08750759156082491, -0.08692334941006566, -0.08624087785649603, 0.0};
double scale_val[5] = {0.0001066894270777766, 0.00010562171582477357, 0.00010899479549851494, 0.00010653598252809886, 6.328949691811795e-07};

// ONNX inputs/outputs
float onnx_input[];
float onnx_output[];

// Feature array size
int lookback = 25;
int features = 5;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Initialize scaler params from JSON (hardcoded for simplicity in MT5,
   // ideally read from scaler_params.json file)
   // We'll leave them as 0s in this placeholder, but in a real scenario
   // you would read the file or embed them.
   // Scaler parameters are initialized globally

   // Load model from resource
   model_handle = OnnxCreateFromBuffer(ExtModelData, ONNX_DEFAULT);
   if(model_handle == INVALID_HANDLE)
     {
      Print("OnnxCreateFromBuffer error ", GetLastError());
      return(INIT_FAILED);
     }

   // Define input/output shapes
   long input_shape[] = {1, lookback, features};
   long output_shape[] = {1, 1};

   if(!OnnxSetInputShape(model_handle, 0, input_shape))
     {
      Print("OnnxSetInputShape error ", GetLastError());
      return(INIT_FAILED);
     }

   if(!OnnxSetOutputShape(model_handle, 0, output_shape))
     {
      Print("OnnxSetOutputShape error ", GetLastError());
      return(INIT_FAILED);
     }

   ArrayResize(onnx_input, lookback * features);
   ArrayResize(onnx_output, 1);

   trade.SetExpertMagicNumber(123456);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(model_handle != INVALID_HANDLE)
     {
      OnnxRelease(model_handle);
      model_handle = INVALID_HANDLE;
     }
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Only run on new bar
   static datetime last_time = 0;
   datetime current_time = iTime(_Symbol, PERIOD_D1, 0);
   if(current_time == last_time) return;

   // Fetch last 25 days data
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, PERIOD_D1, 1, lookback, rates) != lookback) return;

   // Prepare ONNX input
   // Data is in series (index 0 is newest). We need chronological order (index 0 is oldest) for LSTM
   int idx = 0;
   for(int i = lookback - 1; i >= 0; i--)
     {
      onnx_input[idx++] = (float)((rates[i].open - min_val[0]) * scale_val[0]);
      onnx_input[idx++] = (float)((rates[i].high - min_val[1]) * scale_val[1]);
      onnx_input[idx++] = (float)((rates[i].low - min_val[2]) * scale_val[2]);
      onnx_input[idx++] = (float)((rates[i].close - min_val[3]) * scale_val[3]);
      onnx_input[idx++] = (float)((rates[i].tick_volume - min_val[4]) * scale_val[4]);
     }

   // Run inference
   if(!OnnxRun(model_handle, ONNX_NO_CONVERSION, onnx_input, onnx_output))
     {
      Print("OnnxRun error ", GetLastError());
      return;
     }

   double predicted_close_scaled = onnx_output[0];
   double predicted_close = (predicted_close_scaled / scale_val[3]) + min_val[3];

   double current_close = rates[0].close;

   // Market Microstructure Logic (FVG / Consequent Encroachment)
   // A Bullish FVG occurs when Low[0] > High[2] (in standard MT5 0 is current, 1 is prev, 2 is prev-prev)
   // Wait, our rates[] is series, so rates[0] is prev bar, rates[1] is prev-prev bar, rates[2] is prev-prev-prev.
   // Let's use rates[0] (day t-1), rates[1] (day t-2), rates[2] (day t-3).
   // Bullish FVG: rates[0].low > rates[2].high

   bool has_open_positions = PositionsTotal() > 0;
   if(has_open_positions) return; // Only 1 position at a time

   if(rates[0].low > rates[2].high) // Bullish FVG detected
     {
      double fvg_top = rates[0].low;
      double fvg_bottom = rates[2].high;
      double ce = fvg_bottom + (fvg_top - fvg_bottom) / 2.0; // Consequent Encroachment (50%)

      // If LSTM predicts price going up and current price is near CE
      if(predicted_close > current_close)
        {
         double sl = fvg_bottom - (fvg_top - fvg_bottom); // SL below FVG
         double tp = predicted_close;

         // --- 1. Position Sizing & Risk Management (Broker Aligned) ---
            double risk_percent = InpRiskPercent / 100.0;
            double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
            double risk_amount = account_balance * risk_percent;

            // Enforce broker Stops Level (Minimum 50 points away from current price)
            double min_stops_level = 50 * _Point;
            double stop_loss = sl;
            if((current_close - stop_loss) < min_stops_level)
            {
               stop_loss = current_close - min_stops_level;
            }

            // Calculate lot size using Contract Size = 10
            double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
            double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
            double contract_size = 10.0; // Per broker specification image

            double sl_distance_points = (current_close - stop_loss) / _Point;
            double calculated_volume = risk_amount / (sl_distance_points * tick_value * contract_size);

            // Normalize volume to broker limits (Min: 0.01, Max: 50.0, Step: 0.01)
            double min_volume = 0.01;
            double max_volume = 50.0;
            double volume_step = 0.01;

            double final_volume = MathRound(calculated_volume / volume_step) * volume_step;
            if (final_volume < min_volume) final_volume = min_volume;
            if (final_volume > max_volume) final_volume = max_volume;

            // --- 2. Trade Request Formulation ---
            MqlTradeRequest request = {};
            MqlTradeResult  result  = {};

            request.action       = TRADE_ACTION_PENDING; // Buy Limit
            request.symbol       = _Symbol;
            request.volume       = final_volume;
            request.type         = ORDER_TYPE_BUY_LIMIT;
            request.price        = NormalizeDouble(ce, _Digits);
            request.sl           = NormalizeDouble(stop_loss, _Digits);
            request.tp           = NormalizeDouble(predicted_close, _Digits);
            request.deviation    = 10;
            request.magic        = 123456;
            request.comment      = "LSTM_FVG_Contract10";
            request.type_filling = ORDER_FILLING_IOC;

            // --- 3. Order Execution ---
            ResetLastError();
            if(!OrderSend(request, result))
            {
                PrintFormat("Execution Failed | Error Code: %d | Retcode: %d", GetLastError(), result.retcode);
            }
            else
            {
                PrintFormat("Position Executed | Ticket: %d | Volume: %.2f | SL: %.2f | TP: %.2f",
                            result.deal, final_volume, request.sl, request.tp);
            }
        }
     }
   else if(rates[0].high < rates[2].low) // Bearish FVG
     {
      double fvg_bottom = rates[0].high;
      double fvg_top = rates[2].low;
      double ce = fvg_bottom + (fvg_top - fvg_bottom) / 2.0;

      if(predicted_close < current_close)
        {
         double sl = fvg_top + (fvg_top - fvg_bottom);
         double tp = predicted_close;

         // --- 1. Position Sizing & Risk Management (Broker Aligned) ---
            double risk_percent = InpRiskPercent / 100.0;
            double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
            double risk_amount = account_balance * risk_percent;

            // Enforce broker Stops Level (Minimum 50 points away from current price)
            double min_stops_level = 50 * _Point;
            double stop_loss = sl;
            if((stop_loss - current_close) < min_stops_level)
            {
               stop_loss = current_close + min_stops_level;
            }

            // Calculate lot size using Contract Size = 10
            double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
            double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
            double contract_size = 10.0; // Per broker specification image

            double sl_distance_points = (stop_loss - current_close) / _Point;
            double calculated_volume = risk_amount / (sl_distance_points * tick_value * contract_size);

            // Normalize volume to broker limits (Min: 0.01, Max: 50.0, Step: 0.01)
            double min_volume = 0.01;
            double max_volume = 50.0;
            double volume_step = 0.01;

            double final_volume = MathRound(calculated_volume / volume_step) * volume_step;
            if (final_volume < min_volume) final_volume = min_volume;
            if (final_volume > max_volume) final_volume = max_volume;

            // --- 2. Trade Request Formulation ---
            MqlTradeRequest request = {};
            MqlTradeResult  result  = {};

            request.action       = TRADE_ACTION_PENDING; // Sell Limit
            request.symbol       = _Symbol;
            request.volume       = final_volume;
            request.type         = ORDER_TYPE_SELL_LIMIT;
            request.price        = NormalizeDouble(ce, _Digits);
            request.sl           = NormalizeDouble(stop_loss, _Digits);
            request.tp           = NormalizeDouble(predicted_close, _Digits);
            request.deviation    = 10;
            request.magic        = 123456;
            request.comment      = "LSTM_FVG_Contract10";
            request.type_filling = ORDER_FILLING_IOC;

            // --- 3. Order Execution ---
            ResetLastError();
            if(!OrderSend(request, result))
            {
                PrintFormat("Execution Failed | Error Code: %d | Retcode: %d", GetLastError(), result.retcode);
            }
            else
            {
                PrintFormat("Position Executed | Ticket: %d | Volume: %.2f | SL: %.2f | TP: %.2f",
                            result.deal, final_volume, request.sl, request.tp);
            }
        }
     }

   last_time = current_time;
  }
//+------------------------------------------------------------------+
