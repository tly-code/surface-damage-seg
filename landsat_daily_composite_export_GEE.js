/**** Landsat daily composite export ****/
/*
Landsat 8:
  COLLECTION_ID = 'LANDSAT/LC08/C02/T2_TOA'
  BANDS = ['B4', 'B3', 'B2']

Landsat 9:
  COLLECTION_ID = 'LANDSAT/LC09/C02/T2_TOA'
  BANDS = ['B4', 'B3', 'B2']

Landsat 7:
  COLLECTION_ID = 'LANDSAT/LE07/C02/T2_TOA'
  BANDS = ['B3', 'B2', 'B1']
  Note: Landsat 7 images after 2003-05-31 are affected by SLC-off gaps.

To export unmasked images, comment out ".map(maskLandsatClouds)" in Section 5.
*/


/* ------------------------------------------------------------------
   1. ROI
------------------------------------------------------------------ */

var roiRegion = ee.Geometry.Rectangle([
  -111.69132190, -76.38148504,
  -101.92339670, -73.91560727
], null, false);

// Alternative:
// var roi = ee.FeatureCollection(download_shp);
// var roiRegion = roi.geometry();

Map.centerObject(roiRegion, 8);
Map.addLayer(roiRegion, {}, 'ROI');


/* ------------------------------------------------------------------
   2. Parameters
------------------------------------------------------------------ */

var START_DATE = '2022-12-01';
var END_DATE   = '2023-03-01';   // Narrow this date range after suitable acquisition dates are identified.

var COLLECTION_ID = 'LANDSAT/LC08/C02/T2_TOA';
var BANDS = ['B4', 'B3', 'B2'];

var VALID_BAND = BANDS[0];

var CLOUD_THRESHOLD = 80;
var VALID_SCALE = 300;
var TILE_SCALE = 4;

var TOP_K = 30;

var EXPORT_FOLDER = 'Thwaites_download';
var EXPORT_PREFIX = 'LC08_Thwaites_composite_';

var visParam = {
  bands: BANDS,
  min: 0.4,
  max: 1.2
};


/* ------------------------------------------------------------------
   3. Cloud mask
------------------------------------------------------------------ */

var maskLandsatClouds = function(image) {
  var qa = image.select('QA_PIXEL');

  var cloudShadowBitMask = 1 << 4;
  var cloudBitMask = 1 << 3;

  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
    .and(qa.bitwiseAnd(cloudBitMask).eq(0));

  return image.updateMask(mask)
    .copyProperties(image, [
      'system:time_start',
      'system:time_end',
      'system:index'
    ]);
};


/* ------------------------------------------------------------------
   4. Valid-pixel percentage
------------------------------------------------------------------ */

var roiTotalPixelCount = ee.Image.constant(1)
  .clip(roiRegion)
  .reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: roiRegion,
    scale: VALID_SCALE,
    maxPixels: 1e13,
    bestEffort: true,
    tileScale: TILE_SCALE
  })
  .get('constant');

var calculateValidPixelPercentage = function(image) {
  var validPixelCount = image.select(VALID_BAND)
    .reduceRegion({
      reducer: ee.Reducer.count(),
      geometry: roiRegion,
      scale: VALID_SCALE,
      maxPixels: 1e13,
      bestEffort: true,
      tileScale: TILE_SCALE
    })
    .get(VALID_BAND);

  var validPixelPercentage = ee.Number(validPixelCount)
    .divide(ee.Number(roiTotalPixelCount))
    .multiply(100);

  return image.set({
    valid_pixel: validPixelCount,
    valid_pixel_percentage: validPixelPercentage
  });
};


/* ------------------------------------------------------------------
   5. Load Landsat collection
------------------------------------------------------------------ */

var imageCollection = ee.ImageCollection(COLLECTION_ID)
  .filterDate(START_DATE, END_DATE)
  .filterBounds(roiRegion)
  .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_THRESHOLD))
  .sort('CLOUD_COVER', true)
  .map(maskLandsatClouds)   // Comment out this line to export unmasked images.
  .select(BANDS);

print('Number of images:', imageCollection.size());
print('Image names:', imageCollection.aggregate_array('system:index'));


/* ------------------------------------------------------------------
   6. Daily composites
------------------------------------------------------------------ */

var dateList = ee.List(
  imageCollection.aggregate_array('system:time_start')
).map(function(timeStart) {
  return ee.Date(timeStart).format('YYYY-MM-dd');
}).distinct().sort();

var dailyCompositeCollection = ee.ImageCollection.fromImages(
  dateList.map(function(dateString) {
    dateString = ee.String(dateString);

    var dayStart = ee.Date(dateString);
    var dayEnd = dayStart.advance(1, 'day');

    return imageCollection
      .filterDate(dayStart, dayEnd)
      .median()
      .set('system:time_start', dayStart.millis())
      .set('date', dateString)
      .clip(roiRegion);
  })
);

print('Number of daily composites:', dailyCompositeCollection.size());


/* ------------------------------------------------------------------
   7. Sort by valid-pixel percentage
------------------------------------------------------------------ */

var sortedDailyCompositeCollection = dailyCompositeCollection
  .map(calculateValidPixelPercentage)
  .sort('valid_pixel_percentage', false);

print('Top dates:', sortedDailyCompositeCollection.limit(10).aggregate_array('date'));
print('Top valid-pixel percentages:', sortedDailyCompositeCollection.limit(10).aggregate_array('valid_pixel_percentage'));


/* ------------------------------------------------------------------
   8. Export
------------------------------------------------------------------ */

var exportImage = function(image, dateText) {
  Export.image.toDrive({
    image: image.multiply(255).toByte(),
    description: EXPORT_PREFIX + dateText + '_uint8',
    folder: EXPORT_FOLDER,
    fileFormat: 'GeoTIFF',
    crs: 'EPSG:3031',
    region: roiRegion,
    scale: 30,
    maxPixels: 1e13
  });
};

var topCollection = sortedDailyCompositeCollection.limit(TOP_K);
var topList = topCollection.toList(TOP_K);

topCollection.aggregate_array('date').evaluate(function(dateArray) {
  for (var i = 0; i < dateArray.length; i++) {
    var image = ee.Image(topList.get(i));
    var dateString = dateArray[i];

    exportImage(image, dateString);
    Map.addLayer(image, visParam, dateString, false);
  }
});