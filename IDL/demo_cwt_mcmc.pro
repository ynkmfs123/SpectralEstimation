FUNCTION DEMO_CWT_MCMC_PERCENTILE, arr, p
  COMPILE_OPT IDL2

  v = DOUBLE(arr)
  v = REFORM(v, N_ELEMENTS(v), /OVERWRITE)
  v = v[SORT(v)]
  n = N_ELEMENTS(v)

  IF n EQ 1L THEN RETURN, v[0]

  pos = DOUBLE(p) / 100D * DOUBLE(n - 1L)
  lo = LONG(FLOOR(pos))
  hi = LONG(CEIL(pos))
  frac = pos - DOUBLE(lo)

  RETURN, (1D - frac) * v[lo] + frac * v[hi]
END

FUNCTION DEMO_CWT_MCMC_NORMALIZE_SIGNAL, x
  COMPILE_OPT IDL2

  x = DOUBLE(x)
  mu = TOTAL(x, /DOUBLE) / DOUBLE(N_ELEMENTS(x))
  sd = SQRT(TOTAL((x - mu)^2D, /DOUBLE) / DOUBLE(N_ELEMENTS(x)))

  RETURN, (x - mu) / sd
END

FUNCTION DEMO_CWT_MCMC_POW2_TICKS, pmin, pmax
  COMPILE_OPT IDL2

  emin = LONG(CEIL(CWT_MCMC_LOG2(DOUBLE(pmin))))
  emax = LONG(FLOOR(CWT_MCMC_LOG2(DOUBLE(pmax))))
  n = emax - emin + 1L

  IF n LE 0L THEN RETURN, [DOUBLE(pmin), DOUBLE(pmax)]

  RETURN, 2D ^ (DINDGEN(n) + DOUBLE(emin))
END

FUNCTION DEMO_CWT_MCMC_HHMM_LABELS, tick_sec
  COMPILE_OPT IDL2

  n = N_ELEMENTS(tick_sec)
  labels = STRARR(n)

  FOR i = 0L, n - 1L DO BEGIN
    t = LONG(ROUND(DOUBLE(tick_sec[i])))
    hh = (t / 3600L) MOD 24L
    mm = (t MOD 3600L) / 60L
    labels[i] = STRING(hh, FORMAT='(I2.2)') + ':' + STRING(mm, FORMAT='(I2.2)')
  ENDFOR

  RETURN, labels
END

PRO DEMO_CWT_MCMC_BUILD_TIME_TICKS, time_sec, has_real_time, tickv, tickname, xtitle
  COMPILE_OPT IDL2

  time_sec = DOUBLE(time_sec)
  t0 = time_sec[0]
  t1 = time_sec[N_ELEMENTS(time_sec) - 1L]

  IF has_real_time THEN BEGIN
    first_tick = CEIL(t0 / 300D) * 300D
    last_tick = FLOOR(t1 / 300D) * 300D
    n_tick = LONG((last_tick - first_tick) / 300D) + 1L

    IF n_tick LE 0L THEN BEGIN
      tickv = [t0, t1]
    ENDIF ELSE BEGIN
      tickv = first_tick + 300D * DINDGEN(n_tick)
    ENDELSE

    tickname = DEMO_CWT_MCMC_HHMM_LABELS(tickv)
    xtitle = 'Time'
  ENDIF ELSE BEGIN
    n_tick = 6L
    tickv = t0 + (t1 - t0) * FINDGEN(n_tick + 1L) / FLOAT(n_tick)
    tickname = STRTRIM(STRING(LONG(ROUND(tickv - t0))), 2)
    xtitle = 'Time (s)'
  ENDELSE
END

PRO DEMO_CWT_MCMC_LOAD_MATPLOTLIB_COOLWARM, START_INDEX=start_index, N_COLORS=n_colors
  COMPILE_OPT IDL2

  IF N_ELEMENTS(start_index) EQ 0 THEN si = 0L ELSE si = LONG(start_index)
  IF N_ELEMENTS(n_colors) EQ 0 THEN nc = 256L ELSE nc = LONG(n_colors)
  IF si LT 0L THEN si = 0L
  IF nc LT 1L THEN nc = 1L
  IF si + nc GT 256L THEN nc = 256L - si

  r0 = BYTE([ $
    59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, $
    78, 79, 80, 81, 83, 84, 85, 86, 88, 89, 90, 91, 93, 94, 95, 97, $
    98, 99, 100, 102, 103, 104, 106, 107, 108, 110, 111, 112, 114, 115, 117, 118, $
    119, 121, 122, 123, 125, 126, 128, 129, 130, 132, 133, 134, 136, 137, 139, 140, $
    141, 143, 144, 146, 147, 148, 150, 151, 152, 154, 155, 157, 158, 159, 161, 162, $
    163, 165, 166, 167, 169, 170, 171, 173, 174, 175, 177, 178, 179, 181, 182, 183, $
    185, 186, 187, 188, 190, 191, 192, 193, 195, 196, 197, 198, 199, 201, 202, 203, $
    204, 205, 206, 207, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, $
    221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 234, 235, $
    236, 237, 237, 238, 239, 239, 240, 241, 241, 242, 242, 242, 243, 243, 244, 244, $
    245, 245, 245, 245, 246, 246, 246, 247, 247, 247, 247, 247, 247, 247, 247, 247, $
    247, 247, 247, 247, 247, 247, 247, 247, 246, 246, 246, 245, 245, 245, 245, 244, $
    244, 243, 243, 243, 242, 242, 241, 241, 240, 240, 239, 238, 238, 237, 236, 236, $
    235, 234, 233, 233, 232, 231, 230, 229, 228, 227, 227, 226, 225, 224, 223, 222, $
    221, 220, 218, 217, 216, 215, 214, 213, 212, 210, 209, 208, 207, 205, 204, 203, $
    202, 200, 199, 197, 196, 195, 193, 192, 190, 189, 187, 186, 184, 183, 181, 180 $
  ])

  g0 = BYTE([ $
    76, 78, 80, 81, 83, 85, 87, 88, 90, 92, 94, 95, 97, 99, 100, 102, $
    104, 105, 107, 109, 110, 112, 114, 115, 117, 119, 120, 122, 124, 125, 127, 128, $
    130, 132, 133, 135, 136, 138, 139, 141, 143, 144, 146, 147, 149, 150, 151, 153, $
    154, 156, 157, 159, 160, 161, 163, 164, 166, 167, 168, 169, 171, 172, 173, 175, $
    176, 177, 178, 180, 181, 182, 183, 184, 185, 187, 188, 189, 190, 191, 192, 193, $
    194, 195, 196, 197, 198, 199, 200, 201, 201, 202, 203, 204, 205, 205, 206, 207, $
    208, 208, 209, 210, 210, 211, 212, 212, 213, 213, 214, 214, 215, 215, 216, 216, $
    217, 217, 218, 218, 218, 219, 219, 219, 219, 220, 220, 220, 220, 220, 220, 221, $
    220, 220, 219, 219, 218, 218, 217, 217, 216, 215, 215, 214, 213, 213, 212, 211, $
    211, 210, 209, 208, 207, 206, 205, 205, 204, 203, 202, 201, 200, 199, 198, 197, $
    196, 194, 193, 192, 191, 190, 189, 188, 186, 185, 184, 183, 181, 180, 179, 177, $
    176, 175, 173, 172, 170, 169, 168, 166, 165, 163, 162, 160, 159, 157, 156, 154, $
    152, 151, 149, 148, 146, 144, 143, 141, 139, 138, 136, 134, 132, 131, 129, 127, $
    125, 123, 122, 120, 118, 116, 114, 112, 110, 108, 107, 105, 103, 101, 99, 97, $
    95, 93, 90, 88, 86, 84, 82, 80, 78, 75, 73, 71, 69, 66, 64, 62, $
    59, 56, 54, 51, 48, 46, 43, 40, 36, 31, 27, 22, 18, 13, 9, 4 $
  ])

  b0 = BYTE([ $
    192, 194, 195, 197, 198, 200, 201, 203, 204, 206, 207, 209, 210, 211, 213, 214, $
    216, 217, 218, 219, 221, 222, 223, 224, 225, 227, 228, 229, 230, 231, 232, 233, $
    234, 235, 236, 237, 238, 239, 239, 240, 241, 242, 243, 243, 244, 245, 246, 246, $
    247, 248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 252, 253, 253, 253, 254, $
    254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, $
    254, 254, 254, 254, 253, 253, 253, 253, 252, 252, 252, 251, 251, 250, 250, 249, $
    249, 248, 248, 247, 246, 246, 245, 244, 244, 243, 242, 241, 240, 240, 239, 238, $
    237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 222, 221, $
    220, 219, 217, 216, 214, 213, 211, 210, 209, 207, 206, 204, 203, 201, 200, 198, $
    197, 195, 194, 192, 191, 189, 187, 186, 184, 183, 181, 180, 178, 177, 175, 173, $
    172, 170, 169, 167, 166, 164, 162, 161, 159, 158, 156, 155, 153, 151, 150, 148, $
    147, 145, 144, 142, 140, 139, 137, 136, 134, 133, 131, 129, 128, 126, 125, 123, $
    122, 120, 119, 117, 116, 114, 113, 111, 110, 108, 107, 105, 104, 102, 101, 99, $
    98, 96, 95, 93, 92, 91, 89, 88, 86, 85, 84, 82, 81, 79, 78, 77, $
    75, 74, 73, 71, 70, 69, 68, 66, 65, 64, 63, 61, 60, 59, 58, 56, $
    55, 54, 53, 52, 50, 49, 48, 47, 46, 45, 44, 43, 42, 40, 39, 38 $
  ])

  idx = LINDGEN(nc)

  IF nc EQ 1L THEN BEGIN
    pos = DBLARR(1)
  ENDIF ELSE BEGIN
    pos = DOUBLE(idx) * 255D / DOUBLE(nc - 1L)
  ENDELSE

  lo = LONG(FLOOR(pos))
  hi = LONG(CEIL(pos))
  frac = pos - DOUBLE(lo)

  rr = BYTE(ROUND((1D - frac) * DOUBLE(r0[lo]) + frac * DOUBLE(r0[hi])))
  gg = BYTE(ROUND((1D - frac) * DOUBLE(g0[lo]) + frac * DOUBLE(g0[hi])))
  bb = BYTE(ROUND((1D - frac) * DOUBLE(b0[lo]) + frac * DOUBLE(b0[hi])))

  TVLCT, rr, gg, bb, si
END

PRO DEMO_CWT_MCMC_LOAD_IDL_CWT_COLORTABLE, CWT_CT=cwt_ct
  COMPILE_OPT IDL2

  DEVICE, DECOMPOSED=0

  DEMO_CWT_MCMC_LOAD_MATPLOTLIB_COOLWARM, START_INDEX=1L, N_COLORS=240L

  TVLCT, 128B, 128B, 128B, 241
  TVLCT, 31B, 119B, 180B, 242
  TVLCT, 0B, 0B, 0B, 250
  TVLCT, 255B, 255B, 255B, 255

  !P.COLOR = 250
  !P.BACKGROUND = 255
END

PRO DEMO_CWT_MCMC_LOAD_SIGNAL_WITH_TIME, data_file, signal, time_sec, has_real_time
  COMPILE_OPT IDL2

  has_real_time = 1B
  n_read = 0L

  OPENR, lun, data_file, /GET_LUN

  line = ''
  WHILE ~EOF(lun) DO BEGIN
    READF, lun, line
    line = STRTRIM(line, 2)

    IF line EQ '' THEN CONTINUE

    parts = STRSPLIT(line, /EXTRACT)
    IF N_ELEMENTS(parts) LT 2 THEN CONTINUE

    time_str = parts[0]
    value = DOUBLE(parts[1])

    tparts = STRSPLIT(time_str, ':', /EXTRACT)
    IF N_ELEMENTS(tparts) GE 3 THEN BEGIN
      hh = LONG(tparts[0])
      mm = LONG(tparts[1])
      ss = DOUBLE(tparts[2])
      t = DOUBLE(hh) * 3600D + DOUBLE(mm) * 60D + ss
    ENDIF ELSE BEGIN
      t = DOUBLE(n_read)
      has_real_time = 0B
    ENDELSE

    IF n_read EQ 0L THEN BEGIN
      signal = [value]
      time_sec = [t]
    ENDIF ELSE BEGIN

      signal = [signal, value]
      time_sec = [time_sec, t]
    ENDELSE

    n_read += 1L
  ENDWHILE

  FREE_LUN, lun

  IF n_read EQ 0L THEN MESSAGE, 'No valid data were read from ' + data_file
END

PRO DEMO_CWT_MCMC_DRAW_COI_HATCH, time_x, ylog_coi, ylog_bottom
  COMPILE_OPT IDL2

  n = N_ELEMENTS(time_x)
  IF n LT 2L THEN RETURN

  xpoly = [DOUBLE(time_x), REVERSE(DOUBLE(time_x))]
  ypoly = [DOUBLE(ylog_coi), REPLICATE(DOUBLE(ylog_bottom), n)]

  CATCH, err
  IF err EQ 0 THEN BEGIN
    POLYFILL, xpoly, ypoly, /DATA, LINE_FILL=1, ORIENTATION=45, $
              SPACING=0.12, COLOR=241
    POLYFILL, xpoly, ypoly, /DATA, LINE_FILL=1, ORIENTATION=135, $
              SPACING=0.12, COLOR=241
  ENDIF
  CATCH, /CANCEL
END

PRO DEMO_CWT_MCMC_PLOT_RESULTS_FINAL, time_sec, signal, has_real_time, $
                                      periods, power, coi, bg_spectra, factor, $
                                      CWT_CT=cwt_ct
  COMPILE_OPT IDL2


  axis_charsize = 1.25
  label_charsize = 1.15
  panel_charsize = 1.20
  charthick = 1.3
  
  time_x = DOUBLE(time_sec)
  DEMO_CWT_MCMC_BUILD_TIME_TICKS, time_x, has_real_time, xtickv, xtickname, xtitle

  WINDOW, 0, XSIZE=1700, YSIZE=380, TITLE='CWT - MCMC'

  DEMO_CWT_MCMC_LOAD_IDL_CWT_COLORTABLE, CWT_CT=cwt_ct
  ERASE, 255

  black = 250
  gray = 241
  mpl_blue = 242

  left_pos = [0.055, 0.18, 0.455, 0.88]
  !P.POSITION = left_pos

  signal_absmax = MAX(ABS(DOUBLE(signal)))

  IF signal_absmax GT 0D THEN BEGIN
    sig_exp = LONG(FLOOR(ALOG10(signal_absmax)))
  ENDIF ELSE BEGIN
    sig_exp = 0L
  ENDELSE

  sig_scale = 10D ^ DOUBLE(sig_exp)
  sig_plot = DOUBLE(signal) / sig_scale

  ymin = MIN(sig_plot)
  ymax = MAX(sig_plot)
  yrange_signal = [ymin, ymax]

  IF ymax GT ymin THEN BEGIN
    pad = 0.06D * (ymax - ymin)
    yrange_signal = [ymin - pad, ymax + pad]
  ENDIF

  PLOT, time_x, sig_plot, /NODATA, XTITLE=xtitle, YTITLE='Intensity', $
    XSTYLE=1, YSTYLE=1, XRANGE=[time_x[0], time_x[N_ELEMENTS(time_x)-1L]], $
    YRANGE=yrange_signal, XTICKV=xtickv, XTICKS=N_ELEMENTS(xtickv)-1, $
    XTICKNAME=xtickname, XMINOR=1, YMINOR=1, $
    COLOR=black, BACKGROUND=255, $
    CHARSIZE=axis_charsize, CHARTHICK=charthick

  OPLOT, time_x, sig_plot, THICK=2, COLOR=mpl_blue

  IF sig_exp NE 0L THEN BEGIN
    sci_label = 'x10!U' + STRTRIM(STRING(sig_exp), 2) + '!N'

    XYOUTS, left_pos[0], left_pos[3] + 0.028D, sci_label, /NORMAL, $
      CHARSIZE=0.95, CHARTHICK=charthick, COLOR=black
  ENDIF

  XYOUTS, left_pos[0] + 0.02 * (left_pos[2] - left_pos[0]), $
          left_pos[1] + 0.02 * (left_pos[3] - left_pos[1]), $
          '(a) Signal', /NORMAL, $
          CHARSIZE=panel_charsize, CHARTHICK=charthick, COLOR=black

  eps = 1D-300
  threshold95 = (DOUBLE(bg_spectra) * DOUBLE(factor)) > eps
  power_safe = DOUBLE(power) > eps

  sig_score = CWT_MCMC_LOG2(power_safe / threshold95)

  sig_flat = REFORM(sig_score, N_ELEMENTS(sig_score))
  finite_idx = WHERE(FINITE(sig_flat), n_finite)

  IF n_finite GT 0L THEN BEGIN
    valid_vals = sig_flat[finite_idx]
  ENDIF ELSE BEGIN
    valid_vals = [0D]
  ENDELSE

  vmin = DEMO_CWT_MCMC_PERCENTILE(valid_vals, 2D)
  vmax = DEMO_CWT_MCMC_PERCENTILE(valid_vals, 98D)

  color_center = -1.3D
  vmin = MIN([vmin, color_center - 1D-6])
  vmax = MAX([vmax, color_center + 1D-6])

  plot_score = (sig_score > vmin) < vmax

  levels = vmin + (vmax - vmin) * FINDGEN(256) / 255D

  ticks_2pow = DEMO_CWT_MCMC_POW2_TICKS(MIN(periods), MAX(periods))
  ylog = CWT_MCMC_LOG2(periods)
  ytickv = CWT_MCMC_LOG2(ticks_2pow)
  ytickname = STRTRIM(STRING(LONG(ticks_2pow)), 2)

  coi_period = (DOUBLE(coi) > MIN(periods)) < MAX(periods)
  ylog_coi = CWT_MCMC_LOG2(coi_period)
  ylog_bottom = CWT_MCMC_LOG2(MAX(periods))

  right_pos = [0.545, 0.18, 0.885, 0.88]
  !P.POSITION = right_pos

  level_norm = DBLARR(N_ELEMENTS(levels))

  low_idx = WHERE(levels LE color_center, n_low)
  IF n_low GT 0L THEN BEGIN
    level_norm[low_idx] = 0.5D * (levels[low_idx] - vmin) / (color_center - vmin)
  ENDIF

  high_idx = WHERE(levels GT color_center, n_high)
  IF n_high GT 0L THEN BEGIN
    level_norm[high_idx] = 0.5D + 0.5D * (levels[high_idx] - color_center) / (vmax - color_center)
  ENDIF

  level_norm = (level_norm > 0D) < 1D
  c_colors = FIX(1 + ROUND(level_norm * 239D))

  CONTOUR, TRANSPOSE(plot_score), time_x, ylog, /FILL, /CELL_FILL, /NOERASE, LEVELS=levels, $
         C_COLORS=c_colors, XTITLE=xtitle, YTITLE='Period (s)', $
         XSTYLE=1, YSTYLE=1, $
         XRANGE=[time_x[0], time_x[N_ELEMENTS(time_x)-1L]], $
         YRANGE=[MAX(ylog), MIN(ylog)], $
         XTICKV=xtickv, XTICKS=N_ELEMENTS(xtickv)-1, XTICKNAME=xtickname, $
         XMINOR=1, $
         YTICKV=ytickv, YTICKS=N_ELEMENTS(ytickv)-1, YTICKNAME=ytickname, $
         YMINOR=1, COLOR=black, BACKGROUND=255, $
         CHARSIZE=axis_charsize, CHARTHICK=charthick

  DEMO_CWT_MCMC_DRAW_COI_HATCH, time_x, ylog_coi, ylog_bottom

  IF (MIN(valid_vals) LE 0D) AND (MAX(valid_vals) GE 0D) THEN BEGIN
    CONTOUR, TRANSPOSE(sig_score), time_x, ylog, /OVERPLOT, $
             LEVELS=[0D], C_COLORS=[black], THICK=2
  ENDIF

  OPLOT, time_x, ylog_coi, LINESTYLE=2, THICK=2, COLOR=black

  cbar_pos = [0.900, 0.18, 0.920, 0.88]

  FOR icb = 0L, N_ELEMENTS(levels) - 2L DO BEGIN
    y0_cb = cbar_pos[1] + (levels[icb] - vmin) / (vmax - vmin) * (cbar_pos[3] - cbar_pos[1])
    y1_cb = cbar_pos[1] + (levels[icb + 1L] - vmin) / (vmax - vmin) * (cbar_pos[3] - cbar_pos[1])
    POLYFILL, [cbar_pos[0], cbar_pos[2], cbar_pos[2], cbar_pos[0]], $
              [y0_cb, y0_cb, y1_cb, y1_cb], $
              /NORMAL, COLOR=c_colors[icb]
  ENDFOR

  PLOTS, [cbar_pos[0], cbar_pos[2], cbar_pos[2], cbar_pos[0], cbar_pos[0]], $
         [cbar_pos[1], cbar_pos[1], cbar_pos[3], cbar_pos[3], cbar_pos[1]], $
         /NORMAL, COLOR=black, THICK=1

  tick_start = LONG(CEIL(vmin))
  tick_end = LONG(FLOOR(vmax))
  tick_span = tick_end - tick_start
  tick_step = MAX([1L, LONG(CEIL(DOUBLE(tick_span) / 8D))])

  IF tick_end GE tick_start THEN BEGIN
    n_cbar_tick = LONG(FLOOR(DOUBLE(tick_end - tick_start) / DOUBLE(tick_step))) + 1L
    cbar_tickv = DOUBLE(tick_start) + DOUBLE(tick_step) * DINDGEN(n_cbar_tick)
  ENDIF ELSE BEGIN
    cbar_tickv = [0D]
  ENDELSE

  IF (vmin LE 0D) AND (vmax GE 0D) THEN BEGIN
    zero_idx = WHERE(ABS(cbar_tickv) LT 1D-12, n_zero)
    IF n_zero EQ 0L THEN cbar_tickv = [cbar_tickv, 0D]
  ENDIF

  cbar_tickv = cbar_tickv[SORT(cbar_tickv)]
  cbar_tickname = STRARR(N_ELEMENTS(cbar_tickv))

  FOR itick = 0L, N_ELEMENTS(cbar_tickv) - 1L DO BEGIN
    t = LONG(ROUND(cbar_tickv[itick]))

    IF t EQ 0L THEN BEGIN
      cbar_tickname[itick] = '1xT95'
    ENDIF ELSE IF t LT 0L THEN BEGIN
      cbar_tickname[itick] = '1/' + STRTRIM(STRING(LONG(ROUND(2D ^ ABS(t)))), 2)
    ENDIF ELSE BEGIN
      cbar_tickname[itick] = STRTRIM(STRING(LONG(ROUND(2D ^ t))), 2) + 'x'
    ENDELSE
  ENDFOR


  FOR itick = 0L, N_ELEMENTS(cbar_tickv) - 1L DO BEGIN
    y_cb = cbar_pos[1] + (cbar_tickv[itick] - vmin) / (vmax - vmin) * (cbar_pos[3] - cbar_pos[1])
    IF (y_cb GE cbar_pos[1]) AND (y_cb LE cbar_pos[3]) THEN BEGIN
      PLOTS, [cbar_pos[2], cbar_pos[2] + 0.006D], [y_cb, y_cb], /NORMAL, $
             COLOR=black, THICK=1
      XYOUTS, cbar_pos[2] + 0.010D, y_cb - 0.008D, cbar_tickname[itick], /NORMAL, $
              CHARSIZE=1.15, CHARTHICK=charthick, COLOR=black
    ENDIF
  ENDFOR

  XYOUTS, cbar_pos[2] + 0.045D, 0.37D * (cbar_pos[1] + cbar_pos[3]), $
          'Power/T95 (log2 scale)', /NORMAL, ORIENTATION=90D, $
          CHARSIZE=1.15, CHARTHICK=charthick, COLOR=black

  XYOUTS, right_pos[0] + 0.02 * (right_pos[2] - right_pos[0]), $
          right_pos[1] + 0.02 * (right_pos[3] - right_pos[1]), $
          '(b) CWT - MCMC', /NORMAL, $
           CHARSIZE=panel_charsize, CHARTHICK=charthick, COLOR=black

  !P.POSITION = [0.0, 0.0, 1.0, 1.0]


END

PRO DEMO_CWT_MCMC_IDL_FINAL, CWT_CT=cwt_ct
  COMPILE_OPT IDL2

  data_file = '../synthetic_signal.txt'
  dt = 2D

  DEMO_CWT_MCMC_LOAD_SIGNAL_WITH_TIME, data_file, signal, time_sec, has_real_time
  x_original = DEMO_CWT_MCMC_NORMALIZE_SIGNAL(signal)

  dj = 1D / 12D
  s0 = 2D * dt
  j = -1L
  f0 = 6D

  PRINT, '[1/3] Computing CWT...'
  cwt_res = CWT_MCMC_CWT(x_original, dt, dj, s0, j, F0=f0)
  power = cwt_res.power
  freqs = cwt_res.freqs
  coi = cwt_res.coi
  periods = cwt_res.periods

  seed = 1234L
  n_iter = 2000L
  burn_frac = 0.2D
  thin = 4L
  signif_level = 0.95D

  mu_anchor = DOUBLE([-8D, -2D, -4D])
  sd_anchor = DOUBLE([2D, 1D, 2D])

  nu_half_t = 3D
  tau_loga = 0.3D
  tau_alpha = 0.1D
  tau_logc = 0.4D

  prop_sig_loga = 0.01D
  prop_sig_alpha = 0.05D
  prop_sig_logc = 0.1D
  prop_sig_logsig = 0.200D

  PRINT, '[2/3] Running MCMC...'
  mcmc_res = CWT_MCMC_MCMC(power, freqs, n_iter, burn_frac, thin, seed, $
                           mu_anchor, sd_anchor, nu_half_t, $
                           tau_loga, tau_alpha, tau_logc, $
                           prop_sig_loga, prop_sig_alpha, prop_sig_logc, prop_sig_logsig, $
                           /SHOW_PROGRESS, PROGRESS_STEP=5L)

  bg_spectra = mcmc_res.bg_spectra

  factor = -ALOG(1D - signif_level)

  PRINT, '[3/3] Plotting final result...'
  DEMO_CWT_MCMC_PLOT_RESULTS_FINAL, time_sec, signal, has_real_time, $
                                  periods, power, coi, bg_spectra, factor, $
                                  CWT_CT=cwt_ct
END

PRO DEMO_CWT_MCMC, CWT_CT=cwt_ct
  COMPILE_OPT IDL2

  DEMO_CWT_MCMC_IDL_FINAL, CWT_CT=cwt_ct
END
