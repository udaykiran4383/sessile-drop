// lib/processing/angle_utils.dart
import 'dart:math' as math;

/// AngleUtils: robust circle fitting (algebraic + RANSAC) and local polynomial tangent
class AngleUtils {
  /// Algebraic circle fit (Kåsa/Taubin-lite) using least squares.
  /// Returns [cx, cy, r] or throws.
  static List<double>? tryCircleFit(List<double> xs, List<double> ys) {
    final n = xs.length;
    if (n < 3) return null;
    final aMat = List.generate(3, (_) => List.filled(3, 0.0));
    final bVec = List.filled(3, 0.0);
    for (int i = 0; i < n; i++) {
      final x = xs[i];
      final y = ys[i];
      final z = x * x + y * y;
      aMat[0][0] += x * x;
      aMat[0][1] += x * y;
      aMat[0][2] += x;
      aMat[1][0] += x * y;
      aMat[1][1] += y * y;
      aMat[1][2] += y;
      aMat[2][0] += x;
      aMat[2][1] += y;
      aMat[2][2] += 1.0;

      bVec[0] += x * z;
      bVec[1] += y * z;
      bVec[2] += z;
    }
    try {
      final sol = _solveLinearSystem(aMat, bVec);
      final D = sol[0], E = sol[1], F = sol[2];
      final cx = D / 2.0;
      final cy = E / 2.0;
      final rSq = F + cx * cx + cy * cy;
      if (!(rSq.isFinite && rSq > 0.0)) return null;
      final r = math.sqrt(rSq);
      return [cx, cy, r];
    } catch (e) {
      return null;
    }
  }

  /// RANSAC wrapper for circle fitting
  static List<double>? ransacCircleFit(List<double> xs, List<double> ys, {int iterations = 100, double inlierThreshold = 2.0}) {
    final n = xs.length;
    if (n < 6) {
      return tryCircleFit(xs, ys);
    }
    final rnd = math.Random();
    int bestInliers = 0;
    List<int> bestInlierIdx = [];
    List<double>? bestCircle;
    for (int it = 0; it < iterations; it++) {
      final i1 = rnd.nextInt(n);
      int i2 = rnd.nextInt(n), i3 = rnd.nextInt(n);
      if (i2 == i1) i2 = (i1 + 1) % n;
      if (i3 == i1 || i3 == i2) i3 = (i2 + 1) % n;
      final sampleXs = [xs[i1], xs[i2], xs[i3]];
      final sampleYs = [ys[i1], ys[i2], ys[i3]];
      try {
        final c = _circumcircle(sampleXs, sampleYs);
        final cx = c[0], cy = c[1], r = c[2];
        if (!r.isFinite) continue;
        final inliers = <int>[];
        for (int k = 0; k < n; k++) {
          final dx = xs[k] - cx;
          final dy = ys[k] - cy;
          final d = (math.sqrt(dx * dx + dy * dy) - r).abs();
          if (d <= inlierThreshold) inliers.add(k);
        }
        if (inliers.length > bestInliers) {
          bestInliers = inliers.length;
          bestInlierIdx = inliers;
          bestCircle = c;
        }
      } catch (_) {
        continue;
      }
    }
    if (bestInliers < 6) {
      return tryCircleFit(xs, ys) ?? bestCircle;
    }
    final inX = bestInlierIdx.map((i) => xs[i]).toList();
    final inY = bestInlierIdx.map((i) => ys[i]).toList();
    return tryCircleFit(inX, inY) ?? bestCircle;
  }

  static List<double> _circumcircle(List<double> xs, List<double> ys) {
    if (xs.length != 3 || ys.length != 3) throw Exception('Need 3 points');
    final x1 = xs[0], y1 = ys[0];
    final x2 = xs[1], y2 = ys[1];
    final x3 = xs[2], y3 = ys[2];
    final a = x1 - x2;
    final b = y1 - y2;
    final c = x1 - x3;
    final d = y1 - y3;
    final e = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0;
    final f = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0;
    final det = a * d - b * c;
    if (det.abs() < 1e-12) throw Exception('Colinear points');
    final cx = (d * e - b * f) / det;
    final cy = (-c * e + a * f) / det;
    final r = math.sqrt((x1 - cx) * (x1 - cx) + (y1 - cy) * (y1 - cy));
    return [cx, cy, r];
  }

  /// Local polynomial fit around contact for tangent computation
  static double polynomialLocalAngle(List<math.Point> points, math.Point contact, List<double> baselineLine, {required bool isLeft}) {
    if (points.isEmpty) return 90.0;
    final xs = points.map((p) => p.x.toDouble()).toList();
    final ys = points.map((p) => p.y.toDouble()).toList();

    final dx = (xs.reduce((a,b) => math.max(a,b)) - xs.reduce((a,b) => math.min(a,b))).abs();
    final dy = (ys.reduce((a,b) => math.max(a,b)) - ys.reduce((a,b) => math.min(a,b))).abs();
    final fitXasFuncOfY = dy > 1.5 * dx;

    List<double> coeffs;
    try {
      if (fitXasFuncOfY) {
        coeffs = _polyfit(ys, xs, 3);
      } else {
        coeffs = _polyfit(xs, ys, 3);
      }
    } catch (e) {
      return 90.0;
    }

    double independent = fitXasFuncOfY ? contact.y.toDouble() : contact.x.toDouble();
    final b = coeffs.length > 1 ? coeffs[1] : 0.0;
    final c = coeffs.length > 2 ? coeffs[2] : 0.0;
    final d = coeffs.length > 3 ? coeffs[3] : 0.0;
    final slope = b + 2.0 * c * independent + 3.0 * d * independent * independent;
    double dy_dx = fitXasFuncOfY ? 1.0 / (slope + 1e-9) : slope;
    final angleRad = math.atan(dy_dx.abs());
    final angleDeg = angleRad * 180.0 / math.pi;
    bool isInterior = (isLeft && dy_dx > 0) || (!isLeft && dy_dx < 0);
    final finalAngle = isInterior ? angleDeg : 180.0 - angleDeg;
    return finalAngle.clamp(0.0, 180.0);
  }

  static List<double> _polyfit(List<double> x, List<double> y, int degree) {
    final n = x.length;
    final m = degree + 1;
    if (n < m) throw Exception('Not enough points for polyfit');
    final ata = List.generate(m, (_) => List.filled(m, 0.0));
    final atb = List.filled(m, 0.0);
    for (int i = 0; i < n; i++) {
      double xp = 1.0;
      for (int j = 0; j < m; j++) {
        final yk = y[i];
        atb[j] += yk * xp;
        double xp2 = 1.0;
        for (int k = 0; k < m; k++) {
          ata[j][k] += xp * xp2;
          xp2 *= x[i];
        }
        xp *= x[i];
      }
    }
    return _solveLinearSystem(ata, atb);
  }

  static List<double> _solveLinearSystem(List<List<double>> A, List<double> b) {
    final n = A.length;
    final M = List.generate(n, (i) => List<double>.from(A[i])..add(b[i]));
    for (int i = 0; i < n; i++) {
      int pivot = i;
      for (int r = i + 1; r < n; r++) {
        if (M[r][i].abs() > M[pivot][i].abs()) pivot = r;
      }
      if (pivot != i) {
        final tmp = M[i];
        M[i] = M[pivot];
        M[pivot] = tmp;
      }
      if (M[i][i].abs() < 1e-12) throw Exception('Singular matrix');
      for (int r = i + 1; r < n; r++) {
        final factor = M[r][i] / M[i][i];
        for (int c = i; c <= n; c++) M[r][c] -= factor * M[i][c];
      }
    }
    final x = List<double>.filled(n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
      double s = 0.0;
      for (int j = i + 1; j < n; j++) s += M[i][j] * x[j];
      x[i] = (M[i][n] - s) / M[i][i];
    }
    return x;
  }
}