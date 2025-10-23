import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_gallery_saver_plus/image_gallery_saver_plus.dart';
import 'dart:io';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';
import 'image_processor.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Water Contact Angle Analyzer',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _image;
  String? _resultText;
  File? _annotatedImage;
  bool _isProcessing = false;
  final ImagePicker _picker = ImagePicker();

  /// Holds the raw numeric results returned by ImageProcessor for CSV/export
  Map<String, dynamic>? _latestResult;

  Future<void> _pickImage(ImageSource source) async {
    setState(() {
      _isProcessing = true;
      _resultText = 'Processing image...';
      _latestResult = null;
    });

    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1200,
        maxHeight: 1200,
        imageQuality: 85,
      );

      if (pickedFile != null) {
        File imageFile = File(pickedFile.path);

        setState(() {
          _image = null;
          _annotatedImage = null;
        });

        var result = await ImageProcessor.processImage(imageFile);

        setState(() {
          _image = imageFile;
          _resultText = result['text'];
          _annotatedImage = result['annotated'];
          _latestResult = result; // store numeric & metadata for CSV
        });
      }
    } catch (e) {
      print('Error picking image: $e');
      setState(() {
        _resultText = 'Error: $e\n\nPlease check permissions and try again.';
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  /// Save annotated image to gallery
  Future<void> _saveAnnotatedImage() async {
    if (_annotatedImage == null) return;

    try {
      final bytes = await _annotatedImage!.readAsBytes();
      final result = await ImageGallerySaverPlus.saveImage(
        Uint8List.fromList(bytes),
        quality: 100,
        name: "contact_angle_result_${DateTime.now().millisecondsSinceEpoch}",
      );

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(result['isSuccess'] == true
              ? "✅ Annotated image saved to gallery!"
              : "❌ Failed to save image."),
        ),
      );
    } catch (e) {
      print("Error saving image: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("❌ Error saving image: $e")),
      );
    }
  }

  /// Export the latest results (single-row) to CSV in app documents directory.
  Future<void> _exportResultsToCSV() async {
    if (_latestResult == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('No results to export.')),
      );
      return;
    }

    try {
      final dir = await getApplicationDocumentsDirectory();
      final filename = 'contact_angle_results_${DateTime.now().millisecondsSinceEpoch}.csv';
      final filePath = '${dir.path}/$filename';

      // Build header and CSV row
      final headers = [
        'timestamp',
        'source_file',
        'angle_deg',
        'angle_scientific',
        'uncertainty_deg',
        'uncertainty_scientific',
        'circle_deg',
        'poly_deg',
        'surface_type',
        'contour_points',
        'baseline_y',
        'annotated_path'
      ];

      String escape(String v) => '"' + v.replaceAll('"', '""') + '"';

      String timestamp = DateTime.now().toIso8601String();
      String sourceFile = _latestResult!['filename'] ?? (_image?.path.split('/').last ?? '');

      double angle = (_latestResult!['angle_numeric'] as num?)?.toDouble() ?? double.nan;
      double unc = (_latestResult!['uncertainty_numeric'] as num?)?.toDouble() ?? double.nan;
      double circ = (_latestResult!['theta_circle'] as num?)?.toDouble() ?? double.nan;
      double poly = (_latestResult!['theta_poly'] as num?)?.toDouble() ?? double.nan;
      String surface = _latestResult!['surface_type'] ?? '';
      int contourCount = _latestResult!['contour_count'] ?? 0;
      double baselineY = (_latestResult!['baseline_y'] as num?)?.toDouble() ?? double.nan;
      String annotatedPath = _latestResult!['annotated_path'] ?? '';

      final headerLine = headers.map(escape).join(',');
      final row = [
        escape(timestamp),
        escape(sourceFile),
        escape(angle.isFinite ? angle.toStringAsFixed(6) : ''),
        escape(angle.isFinite ? angle.toStringAsExponential(3) : ''),
        escape(unc.isFinite ? unc.toStringAsFixed(6) : ''),
        escape(unc.isFinite ? unc.toStringAsExponential(3) : ''),
        escape(circ.isFinite ? circ.toStringAsFixed(6) : ''),
        escape(poly.isFinite ? poly.toStringAsFixed(6) : ''),
        escape(surface),
        escape(contourCount.toString()),
        escape(baselineY.isFinite ? baselineY.toStringAsFixed(2) : ''),
        escape(annotatedPath),
      ].join(',');

      final csv = '$headerLine\n$row\n';

      final file = File(filePath);
      await file.writeAsString(csv);

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('✅ CSV exported to: $filePath')),
      );
    } catch (e) {
      print('Error exporting CSV: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('❌ Failed to export CSV: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        title: Text('Contact Angle Analyzer'),
        backgroundColor: Colors.blue[700],
        foregroundColor: Colors.white,
        elevation: 2,
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Header
            Container(
              padding: EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.blue[200]!),
              ),
              child: Column(
                children: [
                  Icon(Icons.water_drop, size: 48, color: Colors.blue[700]),
                  SizedBox(height: 8),
                  Text(
                    'Sessile Drop Analysis',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.blue[800],
                    ),
                  ),
                  SizedBox(height: 4),
                  Text(
                    'Automated contact angle measurement for surface science',
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.blue[600],
                    ),
                    textAlign: TextAlign.center,
                  ),
                  SizedBox(height: 8),
                  // Show filename when available
                  if (_image != null)
                    Text(
                      'File: ${_image!.path.split('/').last}',
                      style: TextStyle(fontSize: 12, color: Colors.grey[700]),
                    ),
                ],
              ),
            ),
            SizedBox(height: 24),

            // Action Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed:
                        _isProcessing ? null : () => _pickImage(ImageSource.gallery),
                    icon: Icon(Icons.photo_library),
                    label: Text('Gallery'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green[600],
                      foregroundColor: Colors.white,
                      padding: EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed:
                        _isProcessing ? null : () => _pickImage(ImageSource.camera),
                    icon: Icon(Icons.camera_alt),
                    label: Text('Camera'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue[600],
                      foregroundColor: Colors.white,
                      padding: EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 24),

            // Processing Status
            if (_isProcessing)
              Container(
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.orange[50],
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.orange[200]!),
                ),
                child: Row(
                  children: [
                    SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        valueColor:
                            AlwaysStoppedAnimation<Color>(Colors.orange[700]!),
                      ),
                    ),
                    SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Analyzing droplet...\nThis may take 2-5 seconds',
                        style: TextStyle(color: Colors.orange[800]),
                      ),
                    ),
                  ],
                ),
              ),

            // Results Display
            if (!_isProcessing && _resultText != null)
              Container(
                width: double.infinity,
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.1),
                      blurRadius: 8,
                      offset: Offset(0, 2),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Analysis Results',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Colors.grey[800],
                      ),
                    ),
                    SizedBox(height: 12),
                    Text(
                      _resultText!,
                      style: TextStyle(
                        fontSize: 16,
                        height: 1.4,
                        color: Colors.grey[700],
                      ),
                    ),

                    // Scientific display of angle
                    if (_latestResult != null && _latestResult!['angle_numeric'] != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 12.0),
                        child: Text(
                          'Angle (scientific): ${( (_latestResult!['angle_numeric'] as num).toDouble()).toStringAsExponential(3)}°',
                          style: TextStyle(
                              fontSize: 14, fontWeight: FontWeight.w600, color: Colors.grey[800]),
                        ),
                      ),

                    SizedBox(height: 8),

                    // Export CSV button
                    if (_latestResult != null)
                      ElevatedButton.icon(
                        onPressed: _exportResultsToCSV,
                        icon: Icon(Icons.file_download),
                        label: Text('Export Results (CSV)'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.teal[600],
                          foregroundColor: Colors.white,
                          padding: EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                        ),
                      ),
                  ],
                ),
              ),

            // Original Image
            if (_image != null && !_isProcessing)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(height: 24),
                  Text(
                    'Original Image',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.grey[700],
                    ),
                  ),
                  SizedBox(height: 8),
                  Container(
                    height: 250,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey[300]!),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.file(
                        _image!,
                        fit: BoxFit.contain,
                        width: double.infinity,
                      ),
                    ),
                  ),
                ],
              ),

            // Annotated Image + Save Button
            if (_annotatedImage != null && !_isProcessing)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(height: 24),
                  Text(
                    'Annotated Analysis',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.grey[700],
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Green: Droplet rim | Yellow: Baseline | Cyan: Fitted circle | Magenta: Contact points | White: Tangents',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[500],
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                  SizedBox(height: 8),
                  Container(
                    height: 250,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey[300]!),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.file(
                        _annotatedImage!,
                        fit: BoxFit.contain,
                        width: double.infinity,
                      ),
                    ),
                  ),
                  SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _saveAnnotatedImage,
                          icon: Icon(Icons.download),
                          label: Text("Save Annotated Image"),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.purple[600],
                            foregroundColor: Colors.white,
                            padding: EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                          ),
                        ),
                      ),
                      SizedBox(width: 12),
                      // Quick export CSV next to save
                      if (_latestResult != null)
                        SizedBox(
                          width: 160,
                          child: ElevatedButton.icon(
                            onPressed: _exportResultsToCSV,
                            icon: Icon(Icons.insert_drive_file),
                            label: Text('CSV'),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.indigo[600],
                              foregroundColor: Colors.white,
                              padding: EdgeInsets.symmetric(vertical: 14),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(8),
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                ],
              ),

            // Usage Tips
            SizedBox(height: 24),
            Container(
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.amber[50],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.amber[200]!),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '📸 Tips for Best Results',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: Colors.amber[800],
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    '• Use backlit droplet images (silhouette against bright background)\n'
                    '• Ensure droplet is centered and touches clear surface\n'
                    '• Avoid glare and reflections on droplet surface\n'
                    '• Higher resolution images improve accuracy\n'
                    '• Expected angles: 0-180° (0°=complete wetting, 180°=superhydrophobic)',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.amber[700],
                      height: 1.3,
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}