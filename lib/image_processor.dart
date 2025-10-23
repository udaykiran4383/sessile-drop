// lib/image_processor.dart
import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as imglib;
import 'package:path_provider/path_provider.dart';

import 'processing/angle_utils.dart';

class ImageProcessor {
  /// Main entrypoint: takes an image File and returns a result map
  static Future<Map<String, dynamic>> processImage(File imageFile,
      {int upsample = 2, // upsample factor for subpixel-ish precision
      int gaussianRadius = 3,
      double lowThresholdFactor = 0.4,
      double highThresholdFactor = 1.0}) async {
    try {
      final bytes = await imageFile.readAsBytes();
      imglib.Image? src = imglib.decodeImage(bytes);
      if (src == null) {
        return {'text': '❌ Failed to decode image. Try a different file.', 'annotated': null};
      }

      // 1) Upsample for subpixel-like precision
      final int width0 = src.width;
      final int height0 = src.height;
      final int width = (width0 * upsample).round();
      final int height = (height0 * upsample).round();
      imglib.Image img = imglib.copyResize(src, width: width, height: height, interpolation: imglib.Interpolation.cubic);

      // 2) Grayscale & contrast stretch (simple)
      imglib.Image gray = imglib.grayscale(img);
      _contrastStretch(gray);

      // 3) Smooth
      imglib.Image blurred = imglib.gaussianBlur(gray, radius: gaussianRadius);

      // 4) Gradient (Sobel) -> magnitude + angle
      final grad = _sobelMagnitudeAndAngle(blurred);

      // 5) Non-maximum suppression (here we keep mag but could refine)
      final nms = grad.magnitude; // basic pass-through for now

      // 6) Hysteresis thresholding (Canny-like)
      final double median = _median(grad.magnitude);
      final high = (median * highThresholdFactor).clamp(10.0, 255.0);
      final low = (high * lowThresholdFactor).clamp(5.0, high);
  var edges = _hysteresis(nms, grad.magnitude, grad.width, grad.height, low: low, high: high);

  // 6.5) Morphological cleanup to reduce speckle noise (3x3 open+close)
  edges = _morphOpen(edges, grad.width, grad.height, iterations: 1);
  edges = _morphClose(edges, grad.width, grad.height, iterations: 1);

      // 7) Connected components on edges -> keep largest component near center bottom (droplet)
      final Set<int>? comp = _largestComponent(edges, grad.width, grad.height);
      if (comp == null || comp.length < 30) {
        return {'text': '❌ No droplet detected. Try a clearer silhouette or higher resolution image.', 'annotated': null};
      }

      // 8) Extract contour (ordered boundary points)
      final contour = _extractContourFromMask(comp, grad.width, grad.height);

      if (contour.length < 8) {
        return {'text': '❌ Contour too small (${contour.length}).', 'annotated': null};
      }

    // 9) Baseline detection: robust near-horizontal line in bottom region
    final baseline = _detectBaselineRobust(edges, contour, grad.width, grad.height,
      bottomFracStart: 0.45, bottomFracEnd: 0.95, roiWidthPercent: 80,
      slopeMaxDeg: 3, inlierThreshold: 1.5, iterations: 220);

      // baseline: represented as (a,b,c) normalized for ax + by + c = 0 and endpoints (x1,y1,x2,y2)
      final baselineLine = baseline.lineParams;
      final baselineEndpoints = baseline.endpoints;
      final baselineYapprox = baseline.baselineY; // approximate baseline y for CSV/UI

      // 10) Circle fitting (robust): use rim points just above the baseline
      // We first select the outer rim arc (above baseline), then prefer a narrow vertical
      // "rim belt" close to the baseline to anchor the circle where contact occurs.
      // This improves the lower part of the circle and prevents apex bias.
    final cleanedArc = _cleanContourAndSelectArc(
          contour, baseline.lineParams, baseline.baselineY,
          keepAbove: 1.0, minPoints: 20);

      // Prefer a belt 6–32 px above the baseline (scaled by image height),
      // fallback to the full cleaned arc if insufficient points.
      final beltPts = _selectRimBelt(
        cleanedArc,
        baseline.baselineY,
        grad.height,
        minAbovePx: math.max(6.0, grad.height * 0.015),
        maxAbovePx: math.max(18.0, grad.height * 0.06),
      );

      final List<math.Point>? ptsForFit = beltPts.length >= 8 ? beltPts : cleanedArc;
      if (ptsForFit == null || ptsForFit.length < 8) {
        return {'text': '❌ Not enough points above baseline for fitting (${ptsForFit?.length ?? 0}).', 'annotated': null};
      }

      // Strong anchors at the true contact region (optional).
      // We only use anchors if they are consistent with the belt and baseline.
      final _ContactFallback? rawAnchors = _fallbackContactsFromContour(contour, baselineLine);
      final bool anchorsValid = rawAnchors != null && _anchorsAreReasonable(
        rawAnchors!, baselineLine, baseline.baselineY, cleanedArc, beltPts, grad.width, grad.height,
      );
      final _ContactFallback? anchors = anchorsValid ? rawAnchors : null;
      final List<math.Point> fitPts = List<math.Point>.from(ptsForFit);
      if (anchors != null) {
        bool hasL = fitPts.any((p)=> (p.x-anchors.left.x).abs()<=2 && (p.y-anchors.left.y).abs()<=2);
        bool hasR = fitPts.any((p)=> (p.x-anchors.right.x).abs()<=2 && (p.y-anchors.right.y).abs()<=2);
        if (!hasL) fitPts.add(anchors.left);
        if (!hasR) fitPts.add(anchors.right);
      }

      // Build per-point weights emphasizing the closest-to-baseline samples
      final List<double> weights = fitPts.map((p) {
        final dy = (baseline.baselineY - p.y.toDouble()).abs();
        final lo = math.max(6.0, grad.height * 0.015);
        final hi = math.max(18.0, grad.height * 0.06);
        final t = ((dy - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0); // 0 near baseline, 1 far
        return 1.0 + 3.0 * (1.0 - t); // weights in [1,4], higher near baseline
      }).toList();
      // If we added anchors, boost their weight further
      if (anchors != null) {
        for (int i = 0; i < fitPts.length; i++) {
          final p = fitPts[i];
          if ((p.x-anchors.left.x).abs()<=2 && (p.y-anchors.left.y).abs()<=2) weights[i] *= 2.0;
          if ((p.x-anchors.right.x).abs()<=2 && (p.y-anchors.right.y).abs()<=2) weights[i] *= 2.0;
        }
      }

      // convert to two arrays
  final xs = fitPts.map((p) => p.x.toDouble()).toList();
  final ys = fitPts.map((p) => p.y.toDouble()).toList();

      // Two competing fits: (A) robust RANSAC; (B) constrained through contacts.
      final circleA = AngleUtils.ransacCircleFit(xs, ys, iterations: 220, inlierThreshold: 1.8);
      if (circleA == null) {
        return {'text': '❌ Circle fit failed.', 'annotated': null};
      }

      List<double>? circleB;
      if (anchors != null) {
        circleB = _fitCircleThroughContacts1D(
          fitPts, weights, anchors.left, anchors.right,
          preferCenterAboveY: baseline.baselineY,
        );
        // sanity on constrained result (radius range and center plausibility)
        if (circleB != null) {
          final roughR = _roughRadiusFromArc(fitPts);
          final rr = circleB[2] / (roughR + 1e-9);
          final centerOk = circleB[1] < baseline.baselineY + math.max(10.0, grad.height * 0.05);
          if (!(rr.isFinite && rr > 0.5 && rr < 1.8 && centerOk)) {
            circleB = null; // reject crazy constrained fit
          }
        }
      }
      // non-linear refinement to hug the rim precisely (with safety checks)
      // Pick the seed with lower weighted RMS on the rim belt
      double cxSeed = circleA[0], cySeed = circleA[1], rSeed = circleA[2];
      double rmsA = _circleResidualRMSWeighted(fitPts, weights, cxSeed, cySeed, rSeed);
      if (circleB != null) {
        final rmsB = _circleResidualRMSWeighted(fitPts, weights, circleB[0], circleB[1], circleB[2]);
        if (rmsB.isFinite && rmsB <= rmsA * 0.98) { // prefer constrained if clearly better
          cxSeed = circleB[0]; cySeed = circleB[1]; rSeed = circleB[2];
          rmsA = rmsB;
        }
      }

      var cx = cxSeed, cy = cySeed, r = rSeed;
      final rms0 = _circleResidualRMSWeighted(fitPts, weights, cx, cy, r);
      final refined = _refineCircleLeastSquaresWeighted(fitPts, weights, cx, cy, r, iters: 18, maxStep: 1.8);
      final rms1 = _circleResidualRMSWeighted(fitPts, weights, refined[0], refined[1], refined[2]);
      final centerShift = math.sqrt(math.pow(refined[0]-cx,2)+math.pow(refined[1]-cy,2));
      final radiusRatio = (refined[2] / r).abs();
      // Extra guard: refined circle must intersect the baseline near the belt extents
      bool baselineOk = false;
      final inter = _circleBaselineIntersections(refined[0], refined[1], refined[2], baselineLine, grad.width, grad.height);
      if (inter != null && inter.length >= 2) {
  final xsBelt = fitPts.map((p) => p.x.toDouble()).toList()..sort();
        final leftX = xsBelt.first, rightX = xsBelt.last;
        final tol = math.max(6.0, grad.width * 0.01);
        inter.sort((u,v)=>u.x.compareTo(v.x));
        baselineOk = (inter.first.x >= leftX - 3*tol && inter.last.x <= rightX + 3*tol);
      }
      // Tighter acceptance to avoid odd lower-arc behavior
      if (baselineOk && rms1.isFinite && rms1 <= rms0 * 1.03 && centerShift <= r * 0.25 && radiusRatio > 0.85 && radiusRatio < 1.15) {
        cx = refined[0]; cy = refined[1]; r = refined[2];
      } // else keep the seed result

      // 11) Find left/right contact points: intersection of circle and baseline (solve)
      List<math.Point>? contacts = _circleBaselineIntersections(cx, cy, r, baselineLine, grad.width, grad.height);
      if (contacts == null || contacts.length < 2) {
        // fallback: find leftmost/rightmost contour points near baseline
        final fallback = _fallbackContactsFromContour(contour, baselineLine);
        if (fallback == null) {
          return {'text': '❌ Could not compute contact points.', 'annotated': null};
        }
        // ensure two points
        contacts = [fallback.left, fallback.right];
      }

      final leftPt = contacts[0];
      final rightPt = contacts[1];

      // 12) Compute tangent angles at contact points using circle geometry (tangent vector = perp(radius))
      final baselineVec = math.Point(baselineEndpoints[2] - baselineEndpoints[0], baselineEndpoints[3] - baselineEndpoints[1]);
      final leftAngle = _angleBetweenTangentAndBaseline(cx, cy, leftPt, baselineVec);
      final rightAngle = _angleBetweenTangentAndBaseline(cx, cy, rightPt, baselineVec);
      final avgAngle = (leftAngle + rightAngle) / 2.0;

    // 13) Refine rim points (subpixel) and compute robust local-polynomial angles
    final refinedArc = _refineArcSubpixel(cleanedArc, grad, iterations: 1);
    final rimForAngles = refinedArc.isNotEmpty ? refinedArc : cleanedArc;
    final leftLocalAngle = AngleUtils.polynomialLocalAngle(_nearbyPoints(rimForAngles, leftPt, radius: 16), leftPt, baselineLine, isLeft: true);
    final rightLocalAngle = AngleUtils.polynomialLocalAngle(_nearbyPoints(rimForAngles, rightPt, radius: 16), rightPt, baselineLine, isLeft: false);
    final finalAngle = (avgAngle + (leftLocalAngle + rightLocalAngle) / 2.0) / 2.0;

    // 14) Bootstrap uncertainty
      final uncertainty = _bootstrapUncertainty(xs, ys, baselineLine, baselineVec);

    // 15) Prepare nicer visuals: draw the refined arc polyline (light smoothing)
    final arcPolyline = _connectAndSmoothContour(refinedArc.isNotEmpty ? refinedArc : cleanedArc, iterations: 1, closed: false);

      // 16) Annotate image (draw arc polyline, fitted circle, baseline, contact points, tangents)
      // Compute local polynomial tangents for visualization
      final leftPolyTan = AngleUtils.polynomialLocalTangent(_nearbyPoints(rimForAngles, leftPt, radius: 16), leftPt);
      final rightPolyTan = AngleUtils.polynomialLocalTangent(_nearbyPoints(rimForAngles, rightPt, radius: 16), rightPt);

      final annotated = _drawAnnotatedImage(
        img,
        arcPolyline,
        cx,
        cy,
        r,
        baselineEndpoints,
        leftPt,
        rightPt,
        leftPolyTan,
        rightPolyTan,
        upsample,
      );

      // Save annotated as PNG
      final tmp = await getTemporaryDirectory();
      final outPath = '${tmp.path}/contact_angle_${DateTime.now().millisecondsSinceEpoch}.png';
      final outFile = File(outPath);
      await outFile.writeAsBytes(imglib.encodePng(annotated));

      String surfaceType;
      if (finalAngle < 90) surfaceType = 'Hydrophilic';
      else if (finalAngle < 150) surfaceType = 'Hydrophobic';
      else surfaceType = 'Superhydrophobic';

      String resultText = '''🎯 Contact Angle: ${finalAngle.toStringAsFixed(2)}° ± ${uncertainty.toStringAsFixed(2)}°\n
Methods:
 • Circle-fit (robust): ${avgAngle.toStringAsFixed(2)}°
 • Local-polynomial: ${( (leftLocalAngle + rightLocalAngle) / 2.0).toStringAsFixed(2)}°
 • Final (avg of methods): ${finalAngle.toStringAsFixed(2)}°
Quality:
 • Contour points: ${contour.length}
 • Baseline Y (approx): ${baselineYapprox.toStringAsFixed(2)}
''';

      return {
        'text': resultText,
        'annotated': outFile,
        'annotated_path': outPath,
        'angle_numeric': finalAngle,
        'uncertainty_numeric': uncertainty,
        'theta_circle': avgAngle,
        'theta_poly': (leftLocalAngle + rightLocalAngle) / 2.0,
        'contour_count': contour.length,
        'baseline_y': baselineYapprox,
        'filename': imageFile.path.split(Platform.pathSeparator).last,
        'surface_type': surfaceType,
      };
    } catch (e, st) {
      print('Error in processImage: $e\n$st');
      return {
        'text':
            '❌ Processing failed: ${e.toString()}\nTry better contrast, crop tightly around the droplet, and use a silhouette image.',
        'annotated': null
      };
    }
  }

  // -------------------------
  // Helper methods
  // -------------------------

  // Accepts either an int pixel or imglib.Pixel and returns the red channel (0..255).
  static int _getRedFromPixel(dynamic pixel) {
    if (pixel == null) return 0;
    // int representation (older image versions)
    if (pixel is int) return (pixel >> 16) & 0xFF;
    // imglib.Pixel type (newer versions)
    try {
      // most Pixel implementations expose .r or .red
      final dyn = pixel;
      if (dyn is imglib.Pixel) {
        // Pixel.r is available in image package Pixel
        return dyn.r.toInt();
      }
      // attempt common properties
      final rProp = (pixel as dynamic).r;
      if (rProp is int) return rProp;
      final redProp = (pixel as dynamic).red;
      if (redProp is int) return redProp;
      // fallback: try ARGB int value accessors
      final value = (pixel as dynamic).value;
      if (value is int) return (value >> 16) & 0xFF;
    } catch (_) {}
    // ultimate fallback
    try {
      final val = pixel.toInt();
      return (val >> 16) & 0xFF;
    } catch (_) {}
    return 0;
  }

  static void _contrastStretch(imglib.Image img) {
    int w = img.width, h = img.height;
    int minV = 255, maxV = 0;
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final v = _getRedFromPixel(img.getPixel(x, y));
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
    final range = math.max(1, maxV - minV);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final p = _getRedFromPixel(img.getPixel(x, y));
        final stretched = ((p - minV) * 255 / range).clamp(0, 255).toInt();
        img.setPixelRgba(x, y, stretched, stretched, stretched, 255);
      }
    }
  }

  // Sobel magnitude + angle
  static _GradResult _sobelMagnitudeAndAngle(imglib.Image gray) {
    final w = gray.width, h = gray.height;
    final mag = List<double>.filled(w * h, 0.0);
    final ang = List<double>.filled(w * h, 0.0);

    final gxKernel = [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
    ];
    final gyKernel = [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1]
    ];

    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        double gx = 0.0, gy = 0.0;
        for (int ky = -1; ky <= 1; ky++) {
          for (int kx = -1; kx <= 1; kx++) {
            final v = _getRedFromPixel(gray.getPixel(x + kx, y + ky)).toDouble();
            gx += gxKernel[ky + 1][kx + 1] * v;
            gy += gyKernel[ky + 1][kx + 1] * v;
          }
        }
        final idx = y * w + x;
        mag[idx] = math.sqrt(gx * gx + gy * gy);
        ang[idx] = math.atan2(gy, gx); // radians
      }
    }
    return _GradResult(magnitude: mag, angle: ang, width: w, height: h);
  }

  // Hysteresis: hi/lo thresholds -> produce binary edge mask
  static List<int> _hysteresis(List<double> nms, List<double> mag, int w, int h, {required double low, required double high}) {
    final int n = mag.length;
    final out = List<int>.filled(n, 0);
    for (int i = 0; i < n; i++) {
      if (mag[i] >= high) out[i] = 2; // strong
      else if (mag[i] >= low) out[i] = 1; // weak
      else out[i] = 0;
    }
    // link weak to strong neighbors (8-neighborhood)
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        final idx = y * w + x;
        if (out[idx] == 1) {
          bool connected = false;
          for (int ny = y - 1; ny <= y + 1 && !connected; ny++) {
            for (int nx = x - 1; nx <= x + 1; nx++) {
              final nIdx = ny * w + nx;
              if (out[nIdx] == 2) {
                connected = true;
                break;
              }
            }
          }
          if (connected) out[idx] = 1; // keep weak as edge
          else out[idx] = 0; // discard weak
        } else if (out[idx] == 2) {
          out[idx] = 1; // mark strong as edge (1)
        }
      }
    }
    return out;
  }

  // Connected component largest component (returns mask as set of indices set to inlier)
  static Set<int>? _largestComponent(List<int> binaryMask, int w, int h) {
    final n = binaryMask.length;
    if (w * h != n) return null;
    final visited = List<int>.filled(n, 0);
    int bestSize = 0;
    Set<int>? bestSet;

    for (int idx = 0; idx < n; idx++) {
      if (binaryMask[idx] == 1 && visited[idx] == 0) {
        final queue = <int>[];
        final curSet = <int>{};
        queue.add(idx);
        visited[idx] = 1;
        while (queue.isNotEmpty) {
          final cur = queue.removeLast();
          curSet.add(cur);
          final cy = cur ~/ w;
          final cx = cur % w;
          for (int ny = cy - 1; ny <= cy + 1; ny++) {
            for (int nx = cx - 1; nx <= cx + 1; nx++) {
              if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
              final nIdx = ny * w + nx;
              if (binaryMask[nIdx] == 1 && visited[nIdx] == 0) {
                visited[nIdx] = 1;
                queue.add(nIdx);
              }
            }
          }
        }
        if (curSet.length > bestSize) {
          bestSize = curSet.length;
          bestSet = curSet;
        }
      }
    }
    return bestSet;
  }

  // Build ordered contour from mask indices
  static List<math.Point> _extractContourFromMask(Set<int> maskIndices, int w, int h) {
    final points = <math.Point>[];
    int minx = w, miny = h, maxx = 0, maxy = 0;
    for (final idx in maskIndices) {
      final x = idx % w;
      final y = idx ~/ w;
      minx = math.min(minx, x);
      miny = math.min(miny, y);
      maxx = math.max(maxx, x);
      maxy = math.max(maxy, y);
      bool boundary = false;
      for (int ny = y - 1; ny <= y + 1 && !boundary; ny++) {
        for (int nx = x - 1; nx <= x + 1; nx++) {
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
            boundary = true;
            break;
          }
          final nIdx = ny * w + nx;
          if (!maskIndices.contains(nIdx)) {
            boundary = true;
            break;
          }
        }
      }
      if (boundary) points.add(math.Point(x.toDouble(), y.toDouble()));
    }
    if (points.isEmpty) return points;
    final cx = points.map((p) => p.x).reduce((a, b) => a + b) / points.length;
    final cy = points.map((p) => p.y).reduce((a, b) => a + b) / points.length;
    points.sort((a, b) {
      final ta = math.atan2(a.y - cy, a.x - cx);
      final tb = math.atan2(b.y - cy, b.x - cx);
      return ta.compareTo(tb);
    });
    return points;
  }

  // Baseline detection (RANSAC line) using bottom region edges/points
  // Robust baseline detection using vertical-profile + subpixel refinement
  static _BaselineResult _detectBaselineFromVerticalProfile(List<int> edgesMask, List<math.Point> contour, int w, int h,
      {int roiWidthPercent = 50, int stepX = 4, double subpixelWindow = 3.0}) {
    // 1) Choose a horizontal central ROI (avoid left/right boundaries where droplet may not be centered)
    final int cx = (w / 2).round();
    final int halfRoi = ((w * roiWidthPercent / 100) / 2).round();
    final int x0 = (cx - halfRoi).clamp(0, w - 1);
    final int x1 = (cx + halfRoi).clamp(0, w - 1);

    // 2) For each sampled x in ROI, compute vertical gradient (simple difference) and find strongest downward transition
    final List<double> xSamples = [];
    final List<double> yCandidates = [];

    for (int xs = x0; xs <= x1; xs += stepX) {
      // compute vertical gradient along column xs
      double bestMag = 0.0;
      int bestY = -1;
      for (int y = 2; y < h - 2; y++) {
        final int idxUp = (y - 1) * w + xs;
        final int idxDown = (y + 1) * w + xs;
        // gradient approximate: value_down - value_up
        final valUp = _pixelIntensityFromMask(edgesMask, idxUp);
        final valDown = _pixelIntensityFromMask(edgesMask, idxDown);
        final double g = (valDown - valUp).abs();
        if (g > bestMag) {
          bestMag = g;
          bestY = y;
        }
      }

      if (bestY >= 1) {
        // refine subpixel via quadratic interpolation of vertical magnitudes around bestY
        final int yM = bestY;
        final double m1 = _verticalEdgeMagnitude(edgesMask, xs, yM - 1, w, h);
        final double m2 = _verticalEdgeMagnitude(edgesMask, xs, yM, w, h);
        final double m3 = _verticalEdgeMagnitude(edgesMask, xs, yM + 1, w, h);
        // quadratic vertex offset: delta = 0.5*(m1 - m3)/(m1 - 2*m2 + m3)
        final denom = (m1 - 2.0 * m2 + m3);
        double delta = 0.0;
        if (denom.abs() > 1e-6) delta = 0.5 * (m1 - m3) / denom;
        final double ySub = yM + delta;
        xSamples.add(xs.toDouble());
        yCandidates.add(ySub);
      }
    }

    if (yCandidates.isEmpty) {
      // fallback to bottom row baseline to avoid failure
      return _BaselineResult(lineParams: [0.0, 1.0, - (h - 1).toDouble()], endpoints: [0.0, (h - 1).toDouble(), (w - 1).toDouble(), (h - 1).toDouble()], baselineY: (h - 1).toDouble());
    }

    // 3) Fit a robust line y = m*x + c via least-squares but ignore outliers by trimming extremes
    final int n = xSamples.length;
    final List<int> idxs = List<int>.generate(n, (i) => i);
    // compute mean & std and filter
    final double meanY = yCandidates.reduce((a, b) => a + b) / n;
    final double varY = yCandidates.map((v) => (v - meanY) * (v - meanY)).reduce((a, b) => a + b) / n;
    final double sdY = math.sqrt(varY);
    final List<double> xsFilt = [];
    final List<double> ysFilt = [];
    for (int i = 0; i < n; i++) {
      if ((yCandidates[i] - meanY).abs() <= 1.5 * sdY) {
        xsFilt.add(xSamples[i]);
        ysFilt.add(yCandidates[i]);
      }
    }
    if (xsFilt.length < 3) {
      // not enough after trimming — fallback to using all
      xsFilt.addAll(xSamples);
      ysFilt.addAll(yCandidates);
    }

    // least-squares fit
    double Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    final int mlen = xsFilt.length;
    for (int i = 0; i < mlen; i++) {
      final x = xsFilt[i], y = ysFilt[i];
      Sx += x;
      Sy += y;
      Sxx += x * x;
      Sxy += x * y;
    }
    final denomLS = (mlen * Sxx - Sx * Sx);
    double m = 0.0, c = 0.0;
    if (denomLS.abs() > 1e-9) {
      m = (mlen * Sxy - Sx * Sy) / denomLS;
      c = (Sy - m * Sx) / mlen;
    } else {
      m = 0.0;
      c = ysFilt.reduce((a, b) => a + b) / ysFilt.length;
    }

    // endpoints across full image width
    final double xLeft = 0.0;
    final double yLeft = m * xLeft + c;
    final double xRight = (w - 1).toDouble();
    final double yRight = m * xRight + c;

    // normalized ax + by + cc = 0 (a = m, b = -1)
    final aNorm = m, bNorm = -1.0, cNorm = c;
    final norm = math.sqrt(aNorm * aNorm + bNorm * bNorm);
    final an = aNorm / norm, bn = bNorm / norm, cn = cNorm / norm;

    final baselineY = ((yLeft.clamp(0.0, (h - 1).toDouble()) + yRight.clamp(0.0, (h - 1).toDouble())) / 2.0);

    return _BaselineResult(lineParams: [an, bn, cn], endpoints: [xLeft, yLeft, xRight, yRight], baselineY: baselineY);
  }

  // New robust baseline detector: constrain to near-horizontal lines in bottom region
  static _BaselineResult _detectBaselineRobust(
    List<int> edgesMask,
    List<math.Point> contour,
    int w,
    int h, {
    double bottomFracStart = 0.5, // start of bottom band as fraction of height
    double bottomFracEnd = 0.95,  // end of bottom band as fraction of height
    int roiWidthPercent = 80,     // horizontal central ROI percent
    double slopeMaxDeg = 10.0,    // allow up to +/-10 degrees from horizontal
    double inlierThreshold = 1.5, // max distance in pixels for inliers
    int iterations = 200,
  }) {
    final int yStart = (h * bottomFracStart).clamp(0, h - 1).toInt();
    final int yEnd = (h * bottomFracEnd).clamp(0, h - 1).toInt();
    final int cx = (w / 2).round();
    final int halfRoi = ((w * roiWidthPercent / 100) / 2).round();
    final int x0 = (cx - halfRoi).clamp(0, w - 1);
    final int x1 = (cx + halfRoi).clamp(0, w - 1);

    // Collect candidate edge points in bottom band and central ROI
    final pts = <math.Point<double>>[];
    for (int y = yStart; y <= yEnd; y++) {
      for (int x = x0; x <= x1; x++) {
        if (edgesMask[y * w + x] == 1) {
          pts.add(math.Point(x.toDouble(), y.toDouble()));
        }
      }
    }

    // If no candidates, fallback to vertical-profile method midpoint
    if (pts.length < 10) {
      return _detectBaselineFromVerticalProfile(edgesMask, contour, w, h);
    }

    // Optional: y-histogram to bias toward the dominant horizontal band
    final binSize = 2; // pixels
    final bins = <int, int>{};
    for (final p in pts) {
      final b = (p.y ~/ binSize);
      bins[b] = (bins[b] ?? 0) + 1;
    }
    int bestBin = bins.entries.reduce((a, b) => a.value >= b.value ? a : b).key;
    final double bandY = (bestBin * binSize + binSize / 2).toDouble();

    // Keep points within a vertical window around the dominant band
    final bandTol = 8.0;
    final bandPts = pts.where((p) => (p.y - bandY).abs() <= 3 * bandTol).toList();
    if (bandPts.length >= 20) {
      pts
        ..clear()
        ..addAll(bandPts);
    }

    // RANSAC for near-horizontal line
    final double slopeMax = math.tan(slopeMaxDeg * math.pi / 180.0);
    final rnd = math.Random(12345);
    int bestCount = 0;
    double bestM = 0.0, bestC = (bandY.isFinite ? bandY : (yStart + yEnd) / 2.0);

    for (int it = 0; it < iterations; it++) {
      final p1 = pts[rnd.nextInt(pts.length)];
      final p2 = pts[rnd.nextInt(pts.length)];
      if ((p2.x - p1.x).abs() < 1e-6) continue; // avoid vertical
      final m = (p2.y - p1.y) / (p2.x - p1.x);
      if (m.abs() > slopeMax) continue; // enforce near-horizontal
      final c = p1.y - m * p1.x;

      int count = 0;
      for (final p in pts) {
        final yPred = m * p.x + c;
        final dist = (p.y - yPred).abs();
        if (dist <= inlierThreshold) count++;
      }
      if (count > bestCount) {
        bestCount = count;
        bestM = m;
        bestC = c;
      }
    }

    // If RANSAC found few inliers, fallback to simple horizontal at bandY
    if (bestCount < 10) {
      final y0 = bandY.isFinite ? bandY : ((yStart + yEnd) / 2.0);
      final a = 0.0, b = 1.0, c = -y0;
      final norm = math.sqrt(a * a + b * b);
      return _BaselineResult(
        lineParams: [a / norm, b / norm, c / norm],
        endpoints: [0.0, y0, (w - 1).toDouble(), y0],
        baselineY: y0,
      );
    }

    // Refit least-squares on inliers for stability
    final xs = <double>[];
    final ys = <double>[];
    for (final p in pts) {
      final yPred = bestM * p.x + bestC;
      if ((p.y - yPred).abs() <= inlierThreshold) {
        xs.add(p.x);
        ys.add(p.y);
      }
    }
    double Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    for (int i = 0; i < xs.length; i++) {
      final x = xs[i], y = ys[i];
      Sx += x; Sy += y; Sxx += x * x; Sxy += x * y;
    }
    final denom = xs.length * Sxx - Sx * Sx;
    double m = bestM, c = bestC;
    if (denom.abs() > 1e-9) {
      m = (xs.length * Sxy - Sx * Sy) / denom;
      c = (Sy - m * Sx) / xs.length;
    }

    // Limit slope to slopeMax again just in case
    if (m.abs() > slopeMax) m = m.sign * slopeMax;

    final xLeft = 0.0;
    final xRight = (w - 1).toDouble();
    final yLeft = m * xLeft + c;
    final yRight = m * xRight + c;

    final a = m, b = -1.0, cc = c; // ax + by + c = 0 with b=-1
    final n = math.sqrt(a * a + b * b);
    final an = a / n, bn = b / n, cn = cc / n;
    final baselineY = ((yLeft.clamp(0.0, (h - 1).toDouble()) + yRight.clamp(0.0, (h - 1).toDouble())) / 2.0);
    return _BaselineResult(lineParams: [an, bn, cn], endpoints: [xLeft, yLeft, xRight, yRight], baselineY: baselineY);
  }

  /// helper: magnitude of vertical "edge" at (x,y) using edgesMask (binary)
  static double _verticalEdgeMagnitude(List<int> edgesMask, int x, int y, int w, int h) {
    if (x < 0 || x >= w || y < 1 || y >= h - 1) return 0.0;
    final up = edgesMask[(y - 1) * w + x];
    final mid = edgesMask[y * w + x];
    final down = edgesMask[(y + 1) * w + x];
    // prefer downward transition from bright->dark: use down - up
    return (down - up).toDouble();
  }

  /// helper: for a flat mask we return intensity 0/1 but keep double
  static double _pixelIntensityFromMask(List<int> mask, int idx) {
    if (idx < 0 || idx >= mask.length) return 0.0;
    return mask[idx].toDouble();
  }

  static List<math.Point> _pointsAboveBaseline(List<math.Point> contour, List<double> baseline, {double margin = 1.5}) {
    final a = baseline[0], b = baseline[1], c = baseline[2];
    // estimate baselineY from line using centroid x
    final cx = contour.map((p) => p.x.toDouble()).reduce((a,b)=>a+b)/contour.length;
    final baselineY = - (baseline[0] * cx + baseline[2]) / baseline[1];
    return contour.where((p) => p.y.toDouble() < baselineY - margin).toList();
  }

  // returns cleaned arc points (List<math.Point>) suitable for fitting
  // NOTE: now accepts explicit baselineY to avoid any mismatch with normalized line params.
  static List<math.Point> _cleanContourAndSelectArc(
    List<math.Point> contour,
    List<double> baselineLine,
    double baselineY, { // explicit baseline y in same coords as contour
    double keepAbove = 2.0,
    int minPoints = 20,
  }) {
    if (contour.isEmpty) return <math.Point>[];

    // 1) compute centroid (double)
    final double cx = contour.map((p) => p.x.toDouble()).reduce((a, b) => a + b) / contour.length;
    final double cy = contour.map((p) => p.y.toDouble()).reduce((a, b) => a + b) / contour.length;

    // 2) keep only outer boundary: for each angle keep farthest point from centroid
    final Map<int, math.Point> angleBucket = {};
    for (final p in contour) {
      final double angRad = math.atan2(p.y.toDouble() - cy, p.x.toDouble() - cx);
      final int angDeg = (angRad * 180.0 / math.pi).round();
      final existing = angleBucket[angDeg];
      if (existing == null) {
        angleBucket[angDeg] = p;
      } else {
        final curDist = (p.x.toDouble() - cx) * (p.x.toDouble() - cx) + (p.y.toDouble() - cy) * (p.y.toDouble() - cy);
        final prevDist = (existing.x.toDouble() - cx) * (existing.x.toDouble() - cx) + (existing.y.toDouble() - cy) * (existing.y.toDouble() - cy);
        if (curDist > prevDist) angleBucket[angDeg] = p;
      }
    }
    final outer = angleBucket.values.toList();
    outer.sort((a, b) => math.atan2(a.y - cy, a.x - cx).compareTo(math.atan2(b.y - cy, b.x - cx)));

    if (outer.isEmpty) return <math.Point>[];

    // 3) remove points too close to baseline (we want droplet rim above baseline)
    final a = baselineLine[0], b = baselineLine[1], c = baselineLine[2];
    // baselineY is provided explicitly; use that to avoid confusion with normalized params
    final List<math.Point> filteredStrict = outer.where((p) {
      return p.y.toDouble() < baselineY - keepAbove;
    }).toList();

    // 4) fallback strategies when strict filtering yields too few points
    if (filteredStrict.length >= minPoints) {
      return _smoothArc(filteredStrict, cx, cy);
    }

    // relax: allow points slightly below baseline (keeps more rim for noisy images)
    final List<math.Point> filteredRelax = outer.where((p) {
      return p.y.toDouble() < baselineY + keepAbove; // allow small below baseline
    }).toList();

    if (filteredRelax.length >= minPoints) {
      return _smoothArc(filteredRelax, cx, cy);
    }

    // If still too few, try taking the top half of the outer contour (by polar angle)
    final List<math.Point> topHalf = outer.where((p) {
      final theta = math.atan2(p.y.toDouble() - cy, p.x.toDouble() - cx) * 180.0 / math.pi;
      // top semicircle roughly corresponds to angles between -170..-10 and 10..170 depending orientation;
      // use condition based on y relative to centroid (top points typically have y < cy)
      return p.y.toDouble() < cy + keepAbove;
    }).toList();

    if (topHalf.length >= minPoints) return _smoothArc(topHalf, cx, cy);

    // ultimate fallback: return smoothed outer (even if small) — don't return empty
    return _smoothArc(outer, cx, cy);
  }

  /// Small radial median smoothing helper (keeps angular order)
  static List<math.Point> _smoothArc(List<math.Point> pts, double cx, double cy) {
    if (pts.length <= 6) return pts;
    final rad = pts.map((p) => math.sqrt((p.x.toDouble() - cx) * (p.x.toDouble() - cx) + (p.y.toDouble() - cy) * (p.y.toDouble() - cy))).toList();
    final smoothed = <math.Point>[];
    for (int i = 0; i < pts.length; i++) {
      final prev = rad[(i - 1 + rad.length) % rad.length];
      final cur = rad[i];
      final next = rad[(i + 1) % rad.length];
      final med = [prev, cur, next]..sort();
      final r = med[1];
      final theta = math.atan2(pts[i].y.toDouble() - cy, pts[i].x.toDouble() - cx);
      final px = cx + r * math.cos(theta);
      final py = cy + r * math.sin(theta);
      smoothed.add(math.Point(px, py));
    }
    return smoothed;
  }

  // Robust circle-line intersection using geometric projection.
  // baseline: [a,b,c] for ax + by + c = 0 (may be normalized or not).
  static List<math.Point>? _circleBaselineIntersections(double cx, double cy, double r, List<double> baseline, int w, int h) {
    double a = baseline[0], b = baseline[1], c = baseline[2];

    // normalize coefficients to ensure |(a,b)| = 1 (makes distance computation simple)
    final norm = math.sqrt(a * a + b * b);
    if (norm == 0.0) return null;
    a /= norm; b /= norm; c /= norm;

    // signed distance from center to line (positive on one side, negative on the other)
    final double d = a * cx + b * cy + c; // if a,b,c normalized, |d| is distance

    // if distance greater than radius -> no intersection; allow small tolerance
    final tol = 1e-6;
    if (d.abs() > r + 1e-3) {
      // no real intersection; return null so fallback logic can run
      return null;
    }

    // foot of perpendicular from center to baseline
    final double fx = cx - a * d;
    final double fy = cy - b * d;

    // tangent direction (unit): rotate normal by 90deg -> t = (-b, a)
    final double tx = -b;
    final double ty = a;
    final double lenT = math.sqrt(tx * tx + ty * ty);
    final double ux = tx / (lenT == 0.0 ? 1.0 : lenT);
    final double uy = ty / (lenT == 0.0 ? 1.0 : lenT);

    // distance along tangent from foot to intersection
    double inside = r * r - d * d;
    if (inside < 0 && inside > -1e-6) inside = 0.0; // numerical tolerance
    if (inside < 0) {
      // numeric negative -> treat as no intersection
      return null;
    }

    final double delta = math.sqrt(inside);
    final p1x = fx + ux * delta;
    final p1y = fy + uy * delta;
    final p2x = fx - ux * delta;
    final p2y = fy - uy * delta;

    final List<math.Point> pts = [math.Point(p1x, p1y), math.Point(p2x, p2y)];
    // keep only points inside image bounds (small tolerance)
    final eps = 2.0;
    final filtered = pts.where((p) =>
        p.x >= -eps && p.x <= (w - 1) + eps && p.y >= -eps && p.y <= (h - 1) + eps).toList();

    if (filtered.isEmpty) {
      // If intersection exists but both are slightly outside image, return the original two (so caller can clamp)
      return pts..sort((u, v) => u.x.compareTo(v.x));
    }

    // sort by x (left, right)
    filtered.sort((u, v) => u.x.compareTo(v.x));
    return filtered;
  }

  // fallback method: pick leftmost/rightmost contour points near the baseline.
  // Uses adaptive tolerance relative to image height.
  static _ContactFallback? _fallbackContactsFromContour(List<math.Point> contour, List<double> baseline) {
    if (contour.isEmpty) return null;

    // normalize baseline (safety)
    double a = baseline[0], b = baseline[1], c = baseline[2];
    final nrm = math.sqrt(a * a + b * b);
    if (nrm == 0) return null;
    a /= nrm; b /= nrm; c /= nrm;

    // compute signed distance of each contour point to baseline and keep those close to baseline
    final int N = contour.length;
    final List<_Proj> proj = [];
    for (int i = 0; i < N; i++) {
      final p = contour[i];
      final double d = (a * p.x + b * p.y + c).abs().toDouble(); // distance to baseline
      proj.add(_Proj(point: p, dist: d));
    }

    // adaptive threshold: a small fraction of image height
    final double maxY = contour.map((p) => p.y.toDouble()).reduce((a, b) => a > b ? a : b);
    final double minY = contour.map((p) => p.y.toDouble()).reduce((a, b) => a < b ? a : b);
    final double imgHeightEstimate = (maxY - minY).abs();
    final double tol = math.max(3.0, imgHeightEstimate * 0.01).toDouble(); // at least 3 px or 1% of height

    // keep candidates sorted by x
    final candidates = proj.where((p) => p.dist <= tol).map((p) => p.point).toList();
    if (candidates.isEmpty) {
      // If none inside strict tol, relax to larger tol
  final double tol2 = math.max(8.0, imgHeightEstimate * 0.03).toDouble();
  final candidates2 = proj.where((p) => p.dist <= tol2).map((p) => p.point).toList();
      if (candidates2.isEmpty) {
        // last resort: take global leftmost and rightmost contour points
        final sortedAll = List<math.Point>.from(contour)..sort((u, v) => u.x.compareTo(v.x));
        return _ContactFallback(left: sortedAll.first, right: sortedAll.last);
      } else {
        final sorted = List<math.Point>.from(candidates2)..sort((u, v) => u.x.compareTo(v.x));
        return _ContactFallback(left: sorted.first, right: sorted.last);
      }
    } else {
      final sorted = List<math.Point>.from(candidates)..sort((u, v) => u.x.compareTo(v.x));
      return _ContactFallback(left: sorted.first, right: sorted.last);
    }
  }

  // helper for fallback projection is defined at file level

  static double _angleBetweenTangentAndBaseline(double cx, double cy, math.Point contact, math.Point baselineVec) {
    final rx = contact.x.toDouble() - cx;
    final ry = contact.y.toDouble() - cy;
    final tx = -ry;
    final ty = rx;
    final tvec = math.Point(tx, ty);
    final bvec = baselineVec;
    final dot = (tvec.x * bvec.x + tvec.y * bvec.y);
    final magT = math.sqrt(tvec.x * tvec.x + tvec.y * tvec.y);
    final magB = math.sqrt(bvec.x * bvec.x + bvec.y * bvec.y);
    if (magT == 0.0 || magB == 0.0) return double.nan;
    double cosv = dot / (magT * magB);
    cosv = cosv.clamp(-1.0, 1.0);
    double ang = math.acos(cosv) * 180.0 / math.pi;
    return ang;
  }

  static List<math.Point> _nearbyPoints(List<math.Point> contour, math.Point center, {int radius = 12}) {
    final out = <math.Point>[];
    final rr2 = (radius * radius).toDouble();
    for (final p in contour) {
      final dx = p.x.toDouble() - center.x.toDouble();
      final dy = p.y.toDouble() - center.y.toDouble();
      if (dx * dx + dy * dy <= rr2) out.add(p);
    }
    return out;
  }

  // simple median
  static double _median(List<double> a) {
    final b = List<double>.from(a)..sort();
    final n = b.length;
    if (n == 0) return 0.0;
    if (n % 2 == 1) return b[n ~/ 2];
    return 0.5 * (b[n ~/ 2 - 1] + b[n ~/ 2]);
  }

  // bootstrap uncertainty by subsampling points and recomputing angle from circle fit
  static double _bootstrapUncertainty(List<double> xs, List<double> ys, List<double> baselineLine, math.Point baselineVec) {
    final rnd = math.Random();
    final n = xs.length;
    if (n < 6) return 0.0;
    final trials = <double>[];
    for (int t = 0; t < 20; t++) {
      final sampleIdx = List<int>.generate(n, (_) => rnd.nextInt(n));
      final sX = sampleIdx.map((i) => xs[i]).toList();
      final sY = sampleIdx.map((i) => ys[i]).toList();
      final c = AngleUtils.tryCircleFit(sX, sY);
      if (c == null) continue;
      final cx = c[0], cy = c[1], r = c[2];
      final contacts = _circleBaselineIntersections(cx, cy, r, baselineLine, 1000, 1000);
      if (contacts == null || contacts.length < 2) continue;
      final left = contacts[0];
      final right = contacts[1];
      final leftA = _angleBetweenTangentAndBaseline(cx, cy, left, baselineVec);
      final rightA = _angleBetweenTangentAndBaseline(cx, cy, right, baselineVec);
      final avg = (leftA + rightA) / 2.0;
      if (avg.isFinite) trials.add(avg);
    }
    if (trials.length < 2) return 0.0;
    final mean = trials.reduce((a, b) => a + b) / trials.length;
    final varSum = trials.map((t) => (t - mean) * (t - mean)).reduce((a, b) => a + b) / (trials.length - 1);
    final sd = math.sqrt(varSum);
    return 1.96 * sd / math.sqrt(trials.length); // 95% CI
  }

  // draw annotated image (downscale back to original size at end)
  static imglib.Image _drawAnnotatedImage(
    imglib.Image upImg,
    List<math.Point> contour,
    double cx,
    double cy,
    double r,
    List<double> baselineEnds,
    math.Point leftPt,
    math.Point rightPt,
    List<double> leftPolyTan,
    List<double> rightPolyTan,
    int upsample,
  ) {
    final annotated = imglib.Image.from(upImg);
    final w = annotated.width;
    final h = annotated.height;

    // draw connected contour/arc polyline (green). Not closed to avoid chords.
    if (contour.length > 1) {
      _drawPolyline(annotated, contour, r: 0, g: 220, b: 0, closed: false, thickness: 2);
    }

    // draw fitted circle (cyan)
    if (r.isFinite && r > 1) {
      final steps = (2 * math.pi * r).ceil().clamp(64, 2000);
      for (int i = 0; i < steps; i++) {
        final ang = (i / steps) * 2.0 * math.pi;
        final px = (cx + r * math.cos(ang)).round();
        final py = (cy + r * math.sin(ang)).round();
        if (px >= 0 && px < w && py >= 0 && py < h) annotated.setPixelRgba(px, py, 0, 255, 255, 255);
      }
      if (cx.round() >= 0 && cx.round() < w && cy.round() >= 0 && cy.round() < h) annotated.setPixelRgba(cx.round(), cy.round(), 0, 255, 255, 255);
    }

    // baseline (yellow)
    final x1 = baselineEnds[0].round();
    final y1 = baselineEnds[1].round();
    final x2 = baselineEnds[2].round();
    final y2 = baselineEnds[3].round();
  _drawThickLine(annotated, x1, y1, x2, y2, 255, 200, 0, 3);

    // contact points (magenta)
    annotated.setPixelRgba(leftPt.x.round(), leftPt.y.round(), 255, 0, 255, 255);
    annotated.setPixelRgba(rightPt.x.round(), rightPt.y.round(), 255, 0, 255, 255);

  // draw tangents at contact points from circle (white) to make angle reference clear
  _drawTangent(annotated, cx, cy, leftPt, length: 60, r: 255, g: 255, b: 255);
  _drawTangent(annotated, cx, cy, rightPt, length: 60, r: 255, g: 255, b: 255);

    // draw local polynomial tangents (orange) for comparison
    if (leftPolyTan.length >= 2) {
      _drawTangentVector(annotated, leftPt, leftPolyTan[0], leftPolyTan[1], length: 60, r: 255, g: 140, b: 0);
    }
    if (rightPolyTan.length >= 2) {
      _drawTangentVector(annotated, rightPt, rightPolyTan[0], rightPolyTan[1], length: 60, r: 255, g: 140, b: 0);
    }

    // downscale back to original by factor upsample
    if (upsample > 1) {
      final newW = (w / upsample).round();
      final newH = (h / upsample).round();
      return imglib.copyResize(annotated, width: newW, height: newH, interpolation: imglib.Interpolation.cubic);
    }
    return annotated;
  }

  static void _drawLine(imglib.Image img, int x0, int y0, int x1, int y1, int r, int g, int b) {
    int dx = (x1 - x0).abs();
    int sx = x0 < x1 ? 1 : -1;
    int dy = - (y1 - y0).abs();
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int x = x0, y = y0;
    while (true) {
      if (x >= 0 && x < img.width && y >= 0 && y < img.height) {
        img.setPixelRgba(x, y, r, g, b, 255);
      }
      if (x == x1 && y == y1) break;
      int e2 = 2 * err;
      if (e2 >= dy) {
        err += dy;
        x += sx;
      }
      if (e2 <= dx) {
        err += dx;
        y += sy;
      }
    }
  }

  // --- Simple binary morphology on edge masks (values 0/1) ---
  static List<int> _morphErode(List<int> mask, int w, int h) {
    final out = List<int>.filled(mask.length, 0);
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        int keep = 1;
        for (int ny = y - 1; ny <= y + 1 && keep == 1; ny++) {
          for (int nx = x - 1; nx <= x + 1; nx++) {
            if (mask[ny * w + nx] == 0) { keep = 0; break; }
          }
        }
        out[y * w + x] = keep;
      }
    }
    return out;
  }

  static List<int> _morphDilate(List<int> mask, int w, int h) {
    final out = List<int>.filled(mask.length, 0);
    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        int set = 0;
        for (int ny = y - 1; ny <= y + 1 && set == 0; ny++) {
          for (int nx = x - 1; nx <= x + 1; nx++) {
            if (mask[ny * w + nx] == 1) { set = 1; break; }
          }
        }
        out[y * w + x] = set;
      }
    }
    return out;
  }

  static List<int> _morphOpen(List<int> mask, int w, int h, {int iterations = 1}) {
    var cur = mask;
    for (int i = 0; i < iterations; i++) {
      cur = _morphErode(cur, w, h);
      cur = _morphDilate(cur, w, h);
    }
    return cur;
  }

  static List<int> _morphClose(List<int> mask, int w, int h, {int iterations = 1}) {
    var cur = mask;
    for (int i = 0; i < iterations; i++) {
      cur = _morphDilate(cur, w, h);
      cur = _morphErode(cur, w, h);
    }
    return cur;
  }

  // Draw a polyline by connecting successive points (optionally closed)
  static void _drawPolyline(imglib.Image img, List<math.Point> pts, {required int r, required int g, required int b, bool closed = false, int thickness = 1}) {
    if (pts.length < 2) return;
    for (int i = 0; i < pts.length - 1; i++) {
      final p0 = pts[i];
      final p1 = pts[i + 1];
      _drawThickLine(img, p0.x.round(), p0.y.round(), p1.x.round(), p1.y.round(), r, g, b, thickness);
    }
    if (closed) {
      final p0 = pts.last;
      final p1 = pts.first;
      _drawThickLine(img, p0.x.round(), p0.y.round(), p1.x.round(), p1.y.round(), r, g, b, thickness);
    }
  }

  // Draw a tangent line at a point on the circle (perpendicular to radius)
  static void _drawTangent(imglib.Image img, double cx, double cy, math.Point pt, {int length = 50, int r = 255, int g = 255, int b = 255}) {
    final rx = pt.x.toDouble() - cx;
    final ry = pt.y.toDouble() - cy;
    // tangent direction = (-ry, rx)
    double tx = -ry;
    double ty = rx;
    final norm = math.sqrt(tx * tx + ty * ty);
    if (norm < 1e-6) return;
    tx /= norm; ty /= norm;
    final x0 = (pt.x - tx * length / 2).round();
    final y0 = (pt.y - ty * length / 2).round();
    final x1 = (pt.x + tx * length / 2).round();
    final y1 = (pt.y + ty * length / 2).round();
    _drawLine(img, x0, y0, x1, y1, r, g, b);
  }

  // Draw a tangent given a unit direction (ux, uy) at a point
  static void _drawTangentVector(imglib.Image img, math.Point pt, double ux, double uy, {int length = 50, int r = 255, int g = 255, int b = 255}) {
    final n = math.sqrt(ux * ux + uy * uy);
    if (n < 1e-9) return;
    final tx = ux / n;
    final ty = uy / n;
    final x0 = (pt.x - tx * length / 2).round();
    final y0 = (pt.y - ty * length / 2).round();
    final x1 = (pt.x + tx * length / 2).round();
    final y1 = (pt.y + ty * length / 2).round();
    _drawLine(img, x0, y0, x1, y1, r, g, b);
  }

  // Draw a thicker line by stamping neighboring offsets
  static void _drawThickLine(imglib.Image img, int x0, int y0, int x1, int y1, int r, int g, int b, int thickness) {
    final t = thickness.clamp(1, 7);
    if (t == 1) {
      _drawLine(img, x0, y0, x1, y1, r, g, b);
      return;
    }
    final offsets = <math.Point<int>>[];
    final rad = (t - 1) ~/ 2;
    for (int dy = -rad; dy <= rad; dy++) {
      for (int dx = -rad; dx <= rad; dx++) {
        if (dx * dx + dy * dy <= rad * rad + 1) offsets.add(math.Point(dx, dy));
      }
    }
    for (final o in offsets) {
      _drawLine(img, x0 + o.x, y0 + o.y, x1 + o.x, y1 + o.y, r, g, b);
    }
  }

  // Connect points into a smooth closed contour using Chaikin corner cutting
  static List<math.Point> _connectAndSmoothContour(List<math.Point> contour, {int iterations = 2, bool closed = true}) {
    if (contour.length < 3) return contour;
    List<math.Point> current = List<math.Point>.from(contour);
    for (int it = 0; it < iterations; it++) {
      final next = <math.Point>[];
      final last = current.length - 1;
      final end = closed ? last : last - 1; // if open, stop at last-1
      for (int i = 0; i <= end; i++) {
        final p0 = current[i];
        final p1 = (i == last)
            ? (closed ? current[0] : current[i])
            : current[i + 1];
        if (!closed && i == last) {
          // open contour: copy the last point once
          next.add(p0);
          continue;
        }
        final q = math.Point(0.75 * p0.x + 0.25 * p1.x, 0.75 * p0.y + 0.25 * p1.y);
        final r = math.Point(0.25 * p0.x + 0.75 * p1.x, 0.25 * p0.y + 0.75 * p1.y);
        next..add(q)..add(r);
      }
      current = next;
    }
    return current;
  }

  // --- Subpixel rim refinement using gradient magnitude ridge alignment ---
  static List<math.Point> _refineArcSubpixel(List<math.Point> pts, _GradResult grad, {int iterations = 1}) {
    if (pts.isEmpty) return pts;
    final w = grad.width, h = grad.height;
    List<math.Point> cur = pts;
    for (int it = 0; it < iterations; it++) {
      final next = <math.Point>[];
      for (final p in cur) {
        final xi = p.x.clamp(1.0, (w - 2).toDouble()).toDouble();
        final yi = p.y.clamp(1.0, (h - 2).toDouble()).toDouble();
        final ang = grad.angle[yi.round() * w + xi.round()];
        final ux = math.cos(ang);
        final uy = math.sin(ang);
        // sample magnitude at p - u, p, p + u
        final m1 = _bilinearSample(grad.magnitude, w, h, xi - ux, yi - uy);
        final m2 = _bilinearSample(grad.magnitude, w, h, xi, yi);
        final m3 = _bilinearSample(grad.magnitude, w, h, xi + ux, yi + uy);
        final denom = (m1 - 2*m2 + m3);
        double delta = 0.0;
        if (denom.abs() > 1e-6) delta = 0.5 * (m1 - m3) / denom; // vertex of parabola
        delta = delta.clamp(-0.8, 0.8);
        final nx = xi + ux * delta;
        final ny = yi + uy * delta;
        next.add(math.Point(nx, ny));
      }
      cur = next;
    }
    return cur;
  }

  static double _bilinearSample(List<double> img, int w, int h, double x, double y) {
    x = x.clamp(0.0, (w - 1).toDouble()).toDouble();
    y = y.clamp(0.0, (h - 1).toDouble()).toDouble();
    final x0 = x.floor();
    final y0 = y.floor();
    final int x1 = math.min(x0 + 1, w - 1);
    final int y1 = math.min(y0 + 1, h - 1);
    final fx = x - x0;
    final fy = y - y0;
    double v00 = img[y0 * w + x0];
    double v10 = img[y0 * w + x1];
    double v01 = img[y1 * w + x0];
    double v11 = img[y1 * w + x1];
    final v0 = v00 * (1 - fx) + v10 * fx;
    final v1 = v01 * (1 - fx) + v11 * fx;
    return v0 * (1 - fy) + v1 * fy;
  }

  /// Select a narrow vertical belt of rim points just above the baseline.
  /// Keeps points with baselineY - maxAbovePx <= y <= baselineY - minAbovePx.
  /// If minAbovePx >= maxAbovePx, function swaps them safely.
  static List<math.Point> _selectRimBelt(
    List<math.Point> pts,
    double baselineY,
    int imgH, {
    double minAbovePx = 6.0,
    double maxAbovePx = 24.0,
  }) {
    if (pts.isEmpty) return pts;
    double lo = math.min(minAbovePx, maxAbovePx);
    double hi = math.max(minAbovePx, maxAbovePx);
    final out = pts.where((p) {
      final dy = baselineY - p.y.toDouble(); // positive if above baseline
      return dy >= lo && dy <= hi;
    }).toList();
    // Keep angular order
    return out;
  }

  // Non-linear least squares refinement for circle parameters minimizing
  // radial residuals e_i = sqrt((x-cx)^2+(y-cy)^2) - r using Gauss-Newton.
  static List<double> _refineCircleLeastSquares(List<math.Point> pts, double cx0, double cy0, double r0, {int iters = 8, double maxStep = 5.0}) {
    double cx = cx0, cy = cy0, r = r0;
  for (int it = 0; it < iters; it++) {
      double a11 = 0, a12 = 0, a13 = 0;
      double a22 = 0, a23 = 0, a33 = 0;
      double b1 = 0, b2 = 0, b3 = 0;
      int used = 0;
      for (final p in pts) {
        final dx = p.x.toDouble() - cx;
        final dy = p.y.toDouble() - cy;
        final rho = math.sqrt(dx * dx + dy * dy);
        if (rho < 1e-6) continue;
        final e = rho - r; // residual
        final j1 = -dx / rho; // d e / d cx
        final j2 = -dy / rho; // d e / d cy
        final j3 = -1.0;      // d e / d r

        // Accumulate J^T J and -J^T e
        a11 += j1 * j1; a12 += j1 * j2; a13 += j1 * j3;
        a22 += j2 * j2; a23 += j2 * j3; a33 += j3 * j3;
        b1  += -j1 * e; b2  += -j2 * e; b3  += -j3 * e;
        used++;
      }
      if (used < 6) break;

      // Solve symmetric 3x3 system
      final A = [
        [a11, a12, a13],
        [a12, a22, a23],
        [a13, a23, a33],
      ];
      final b = [b1, b2, b3];
      List<double>? delta = _solve3x3(A, b);
      if (delta == null) break;
      if (delta.length != 3 || delta.any((d) => !d.isFinite)) break;
      // clamp step to avoid divergence
      double sx = delta[0], sy = delta[1], sr = delta[2];
      final smax = [sx.abs(), sy.abs(), sr.abs()].reduce((a,b)=>a>b?a:b);
      if (smax > maxStep) {
        final scale = maxStep / (smax + 1e-9);
        sx *= scale; sy *= scale; sr *= scale;
      }
      // simple line search: accept only if residual decreases
      final rmsPrev = _circleResidualRMS(pts, cx, cy, r);
      double bestScale = 1.0;
      List<double> scales = [1.0, 0.5, 0.25];
      bool accepted = false;
      for (final sc in scales) {
        final ncx = cx + sx * sc;
        final ncy = cy + sy * sc;
        final nr  = r  + sr * sc;
        final rmsNew = _circleResidualRMS(pts, ncx, ncy, nr);
        if (rmsNew < rmsPrev) {
          bestScale = sc;
          cx = ncx; cy = ncy; r = nr;
          accepted = true;
          break;
        }
      }
      if (!accepted) break;
      if ((sx*bestScale).abs() + (sy*bestScale).abs() + (sr*bestScale).abs() < 1e-3) break;
    }
    return [cx, cy, r];
  }

  // Weighted version of the non-linear refinement: each residual is scaled
  // by sqrt(w_i). This emphasizes points near the baseline to lock the lower arc.
  static List<double> _refineCircleLeastSquaresWeighted(
    List<math.Point> pts,
    List<double> w,
    double cx0,
    double cy0,
    double r0, {
    int iters = 12,
    double maxStep = 2.0,
  }) {
    double cx = cx0, cy = cy0, r = r0;
    final n = pts.length;
    if (w.length != n) return [cx, cy, r];
    for (int it = 0; it < iters; it++) {
      double a11 = 0, a12 = 0, a13 = 0;
      double a22 = 0, a23 = 0, a33 = 0;
      double b1 = 0, b2 = 0, b3 = 0;
      int used = 0;
      for (int i = 0; i < n; i++) {
        final p = pts[i];
        final wi = math.sqrt(w[i].clamp(1e-6, 1e6));
        final dx = p.x.toDouble() - cx;
        final dy = p.y.toDouble() - cy;
        final rho = math.sqrt(dx * dx + dy * dy);
        if (rho < 1e-6) continue;
        final e = (rho - r) * wi; // weighted residual
        final j1 = (-dx / rho) * wi;
        final j2 = (-dy / rho) * wi;
        final j3 = (-1.0) * wi;

        a11 += j1 * j1; a12 += j1 * j2; a13 += j1 * j3;
        a22 += j2 * j2; a23 += j2 * j3; a33 += j3 * j3;
        b1  += -j1 * e; b2  += -j2 * e; b3  += -j3 * e;
        used++;
      }
      if (used < 6) break;
      final A = [
        [a11, a12, a13],
        [a12, a22, a23],
        [a13, a23, a33],
      ];
      final bb = [b1, b2, b3];
      final delta = _solve3x3(A, bb);
      if (delta == null || delta.any((d) => !d.isFinite)) break;
      double sx = delta[0], sy = delta[1], sr = delta[2];
      final smax = [sx.abs(), sy.abs(), sr.abs()].reduce((a,b)=>a>b?a:b);
      if (smax > maxStep) {
        final scale = maxStep / (smax + 1e-9);
        sx *= scale; sy *= scale; sr *= scale;
      }
      final rmsPrev = _circleResidualRMSWeighted(pts, w, cx, cy, r);
      bool accepted = false; double bestScale = 1.0;
      for (final sc in const [1.0, 0.5, 0.25]) {
        final ncx = cx + sx * sc;
        final ncy = cy + sy * sc;
        final nr  = r  + sr * sc;
        final rmsNew = _circleResidualRMSWeighted(pts, w, ncx, ncy, nr);
        if (rmsNew < rmsPrev) { cx = ncx; cy = ncy; r = nr; bestScale = sc; accepted = true; break; }
      }
      if (!accepted) break;
      if ((sx*bestScale).abs() + (sy*bestScale).abs() + (sr*bestScale).abs() < 5e-4) break;
    }
    return [cx, cy, r];
  }

  static double _circleResidualRMS(List<math.Point> pts, double cx, double cy, double r) {
    if (pts.isEmpty) return double.infinity;
    double ss = 0.0;
    for (final p in pts) {
      final dx = p.x.toDouble() - cx;
      final dy = p.y.toDouble() - cy;
      final e = math.sqrt(dx*dx + dy*dy) - r;
      ss += e*e;
    }
    return math.sqrt(ss / pts.length);
  }

  static double _circleResidualRMSWeighted(List<math.Point> pts, List<double> w, double cx, double cy, double r) {
    if (pts.isEmpty || w.length != pts.length) return double.infinity;
    double ss = 0.0, ws = 0.0;
    for (int i = 0; i < pts.length; i++) {
      final p = pts[i];
      final wi = w[i].clamp(1e-9, 1e9);
      final dx = p.x.toDouble() - cx;
      final dy = p.y.toDouble() - cy;
      final e = math.sqrt(dx*dx + dy*dy) - r;
      ss += wi * e * e; ws += wi;
    }
    return math.sqrt(ss / ws);
  }

  // Estimate a rough radius from arc points using centroid-median distance.
  static double _roughRadiusFromArc(List<math.Point> pts) {
    if (pts.isEmpty) return 1.0;
    final cx = pts.map((p)=>p.x.toDouble()).reduce((a,b)=>a+b)/pts.length;
    final cy = pts.map((p)=>p.y.toDouble()).reduce((a,b)=>a+b)/pts.length;
    final dists = pts.map((p){
      final dx = p.x.toDouble()-cx; final dy = p.y.toDouble()-cy; return math.sqrt(dx*dx+dy*dy);
    }).toList()..sort();
    return dists[dists.length~/2].clamp(1.0, 1e6);
  }

  // Validate that anchor contact points are believable and consistent with the belt region.
  static bool _anchorsAreReasonable(
    _ContactFallback anchors,
    List<double> baselineLine,
    double baselineY,
    List<math.Point> cleanedArc,
    List<math.Point> beltPts,
    int w,
    int h,
  ) {
    if (beltPts.isEmpty) return false;
    // normalize line
    double a = baselineLine[0], b = baselineLine[1], c = baselineLine[2];
    final n = math.sqrt(a*a + b*b);
    if (n == 0) return false; a/=n; b/=n; c/=n;
    double distToBaseline(math.Point p)=> (a*p.x + b*p.y + c).abs();
    final tol = math.max(3.0, h*0.01);
    // each anchor should be near the baseline
    if (distToBaseline(anchors.left) > 2*tol) return false;
    if (distToBaseline(anchors.right) > 2*tol) return false;
    // x-range sanity: anchors should lie close to belt x-span
    final xs = beltPts.map((p)=>p.x.toDouble()).toList()..sort();
    final leftX = xs.first, rightX = xs.last;
    if (anchors.left.x < leftX - 4*tol) return false;
    if (anchors.right.x > rightX + 4*tol) return false;
    // chord length sanity: not too tiny, not absurdly large
    final chord = math.sqrt(math.pow(anchors.right.x-anchors.left.x,2)+math.pow(anchors.right.y-anchors.left.y,2));
    if (chord < math.max(8.0, w*0.03)) return false;
    if (chord > math.max(400.0, w*0.9)) return false;
    return true;
  }

  /// Constrained circle fit passing exactly through the two contact points (left/right).
  /// The circle center is parameterized along the perpendicular bisector of the chord LR.
  /// We search for the best center offset t (>= 0) that minimizes the weighted RMS on pts.
  static List<double> _fitCircleThroughContacts1D(
    List<math.Point> pts,
    List<double> w,
    math.Point left,
    math.Point right, {
    required double preferCenterAboveY,
  }) {
    final mx = (left.x.toDouble() + right.x.toDouble()) * 0.5;
    final my = (left.y.toDouble() + right.y.toDouble()) * 0.5;
    final vx = right.x.toDouble() - left.x.toDouble();
    final vy = right.y.toDouble() - left.y.toDouble();
    final d = math.sqrt(vx * vx + vy * vy);
    if (!(d.isFinite) || d < 1e-6) {
      return [mx, my, 1e9];
    }
    // unit perpendicular to chord
    final ux = -vy / d;
    final uy =  vx / d;

    // choose sign placing center above the baseline (smaller y in image coords)
    final sign = (my - preferCenterAboveY) > 0 ? -1.0 : 1.0; // if midpoint below baseline, go upward (y smaller)

    // golden-section search over t >= 0
    double tLo = 0.0;
    double tHi = math.max(10.0, d * 6.0);
    double gr = (math.sqrt(5) - 1) / 2; // ~0.618

    double f(double t) {
      final cx = mx + ux * (sign * t);
      final cy = my + uy * (sign * t);
      final r = math.sqrt(t * t + (d * 0.5) * (d * 0.5));
      return _circleResidualRMSWeighted(pts, w, cx, cy, r);
    }

    double a = tLo, b = tHi;
    double c = b - gr * (b - a);
    double d2 = a + gr * (b - a);
    double fc = f(c), fd = f(d2);
    for (int it = 0; it < 50; it++) {
      if (fc > fd) {
        a = c; c = d2; fc = fd; d2 = a + gr * (b - a); fd = f(d2);
      } else {
        b = d2; d2 = c; fd = fc; c = b - gr * (b - a); fc = f(c);
      }
    }
    final tBest = (a + b) * 0.5;
    final cx = mx + ux * (sign * tBest);
    final cy = my + uy * (sign * tBest);
    final r = math.sqrt(tBest * tBest + (d * 0.5) * (d * 0.5));
    return [cx, cy, r];
  }

  // tiny 3x3 solver (Gaussian elimination with partial pivoting)
  static List<double>? _solve3x3(List<List<double>> A, List<double> b) {
    final M = [
      [A[0][0], A[0][1], A[0][2], b[0]],
      [A[1][0], A[1][1], A[1][2], b[1]],
      [A[2][0], A[2][1], A[2][2], b[2]],
    ];
    final n = 3;
    for (int i = 0; i < n; i++) {
      int piv = i;
      for (int r = i + 1; r < n; r++) {
        if (M[r][i].abs() > M[piv][i].abs()) piv = r;
      }
      if (M[piv][i].abs() < 1e-12) return null;
      if (piv != i) {
        final tmp = M[i];
        M[i] = M[piv];
        M[piv] = tmp;
      }
      for (int r = i + 1; r < n; r++) {
        final f = M[r][i] / M[i][i];
        for (int c = i; c <= n; c++) {
          M[r][c] -= f * M[i][c];
        }
      }
    }
    final x = List<double>.filled(n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
      double s = 0.0;
      for (int j = i + 1; j < n; j++) s += M[i][j] * x[j];
      final denom = M[i][i];
      if (denom.abs() < 1e-12) return null;
      x[i] = (M[i][n] - s) / denom;
    }
    return x;
  }
}

// small helper types
class _GradResult {
  final List<double> magnitude;
  final List<double> angle;
  final int width;
  final int height;
  _GradResult({required this.magnitude, required this.angle, required this.width, required this.height});
}

class _BaselineResult {
  final List<double> lineParams; // a,b,c normalized
  final List<double> endpoints; // x1,y1,x2,y2 (in upsample coords)
  final double baselineY;
  _BaselineResult({required this.lineParams, required this.endpoints, required this.baselineY});
}

class _ContactFallback {
  final math.Point left;
  final math.Point right;
  _ContactFallback({required this.left, required this.right});
}

// helper for fallback projection (file-level)
class _Proj {
  final math.Point point;
  final double dist;
  _Proj({required this.point, required this.dist});
}
