library(tidyverse)
library(tigris)
library(sf)
library(leaflet)

# -- Given a TIGER file, create intersection points -- #
create_intersection <- function(input, include_deadend = FALSE){
  library(tidyverse)
  # Given the input road centerline shapefile, this function creates intersection points.
  
  # Arguements
  #   input: a line shapefile representing road centerline
  #   include_deadend (bool): FALSE returns only true intersections.
  #                           TRUE may include deadends such as cul-de-sacs.                          
  
  # Return
  #   Point sf object
  
  message("[INFO] Creating intersections using sf package")
  
  if(!any(class(input) %in% c("sf", 'sfc'))){
    stop("Needs sf or sfc objects")
  }
  
  # unique ID
  input$lineID = seq(1, nrow(input))
  
  # Base intersections using sf package
  input.inter = sf::st_intersection(input)
  filter_index = sf::st_geometry_type(input.inter) %in% c("POINT", "MULTIPOINT")
  x = input.inter[filter_index,]
  x = sf::st_cast(x, "MULTIPOINT")
  x = sf::st_cast(x, "POINT")
  
  # Adding missing nodes
  # message("[INFO] Creating additional nodes that are missing from sf package")
  y = data.frame()
  for (i in x$lineID){
    t = sf::st_coordinates(input[input$lineID == as.numeric(i),])
    t2 = t[c(1, nrow(t)), c("X", "Y")]
    y = rbind(y, t2)
  }
  
  # message("[INFO] Merging the two types of intersections")
  z = y %>% 
    rbind(as.data.frame(sf::st_coordinates(x))) %>%
    dplyr::distinct(X,Y) %>% 
    sf::st_as_sf(coords = c('X', 'Y'), dim = "XY", crs = sf::st_crs(input)) %>% 
    sf::st_transform(crs = 26967)
  
  # Deleting deadends
  if (include_deadend == FALSE){
    z$zid = seq(1, nrow(z))
    j = sf::st_join(input %>% sf::st_transform(26967), z %>% sf::st_buffer(units::set_units(1, "m")))
    s = j %>% group_by(zid) %>% summarise(n = n()) %>% filter(!is.na(.$zid))
    not.deadend.index = s$zid[s$n >= 2]
    z = z[z$zid %in% not.deadend.index,]
  }
  
  # Turning it back to 4326
  z <- z %>% sf::st_transform(crs = 4326)
  
  return(z)
}

# -- Break the line at all intersections -- #
break_line_at_point <- function(line, point, clean_line = TRUE){
  # Break all lines at the intersection points
  
  # Arguements
  #   line (sf object, "LINESTRING" or "MULTILINESTRING"): sf object of line shapefile representing street centerline
  #   point (sf object, "POINT", or "MULTIPOINT"): sf object of point shapefile representing intersections
  
  # Return
  #   sf "LINESTRING" object with id column. Each row is 
  #   individual line segment between intersections.
  
  # Inspiration: 
  # https://stackoverflow.com/questions/55519152/split-line-by-multiple-points-using-sf-package
  
  # Combine all sf features
  line_all = sf::st_combine(line) %>% st_transform(26967)
  point_all = sf::st_combine(point) %>% st_transform(26967)
  
  # Buffer the point by small distance
  point_all <- sf::st_snap(point_all,
                           line_all,
                           tolerance = units::set_units(1, "m"))
  
  # Split the line at points and extract all LINESTRINGS
  parts = sf::st_collection_extract(
    lwgeom::st_split(line_all, point_all),
    "LINESTRING")
  
  # Create sf object
  if (clean_line == TRUE){
    parts = sf::st_as_sf(
      data.frame(
        geometry = parts
      ))
    
    parts = sf::st_difference(parts)
    parts$SegID = 1:nrow(parts)
    
  } else {
    parts = sf::st_as_sf(
      data.frame(
        SegID = 1:length(parts),
        geometry = parts
      ))
  }
  
  return(parts)
}

# -- Calculate segment azimuth -- #
get_segAzi <- function(line){
  # Given the input road centerline shapefile, this function calculates
  # the azimuth of each segment (defined by the two intersections)    
  
  # Arguements
  #   line (sf object, "LINESTRING" or "MULTILINESTRING"): a line shapefile representing road centerline.
  
  # Return
  #   data.frame with 2 variables SegID, SegAzi
  
  # Calculate azimuth for each SegID
  seg_azimuth <- map_df(seq(1,nrow(line)), function(x) {
    SegID <- line$SegID[x]
    g <- st_coordinates(line[x,]) %>% as_tibble()
    p1 <- g[1,]
    p2 <- g[nrow(g),]
    
    azi <- atan2(p2$X - p1$X, p2$Y - p1$Y)*180/pi
    
    o <- data.frame(SegID = SegID, SegAzi = azi)
    return(o)    
  })
  
  # Fix azimuth outside [0, 360]
  seg_azimuth <- line %>% 
    left_join(seg_azimuth, by = "SegID") %>% 
    mutate(SegAzi = case_when(
      SegAzi < 0 ~ SegAzi + 360,
      SegAzi > 360 ~ SegAzi - 360,
      SegAzi == 360 ~ 0,
      TRUE ~ SegAzi
    )) %>% 
    select(SegID, SegAzi) %>% 
    st_set_geometry(NULL)
  
  return(seg_azimuth)
}


# -- Break each street segment into 5 meter interval -- #
break_seg_at_interval <- function(input_segment, interval = 5){
  # Given the input road centerline shapefile, break the lines at the given distance (default is 5 meters).
  
  # Arguements
  #   input_segment (sf object, "LINESTRING" or "MULTILINESTRING"): a line shapefile representing road centerline.
  #   interval (numeric): Distance at which to break the line. In meters.
  
  # Return
  #   Point sf object
  
  message("[INFO] Breaking street segments at 5-meter interval to create dense points")
  # Add additional vertices at every given distance
  to_coordinates <- st_segmentize(input_segment, units::set_units(interval, m)) %>% 
    st_cast("MULTILINESTRING") %>% 
    st_coordinates()
  
  segid.match <- data.frame(SegID = input_segment$SegID, id = seq(1:nrow(input_segment)))
  to_points <- to_coordinates %>% 
    as.data.frame() %>% 
    st_as_sf(coords = c("X", "Y"), crs = 26967, dim = "XY") %>% 
    left_join(segid.match, by = c("L2" = "id"))
  
  segmented_lines.list <- map(input_segment$SegID, function(x){
    # get the point coordinates of original segment & segmented segment
    orig <- input_segment %>% filter(SegID == x) %>% st_coordinates() %>% as.data.frame() %>% mutate(XY = paste0(X,Y))
    segmented <- to_points %>% filter(SegID == x) %>% st_coordinates() %>% as.data.frame() %>% mutate(XY = paste0(X,Y))
    # find out which points are added ones (for x)
    if (nrow(orig) >= 2){
      dif <- setdiff(segmented[,'XY'], orig[,'XY'])
      dif_index <- segmented[,'XY'] %in% dif
      
      # locate the index of added points & add 1 and end index
      where_true <- which(dif_index %in% TRUE)
      where_true <- c(1, where_true, nrow(segmented))
      # create index with which to extract segmented
      extract_index <- c()
      
      for (i in 1:length(where_true)){
        if (i + 1 <= length(where_true)){
          extract_index <- c(extract_index, where_true[i]:where_true[i+1])
        }
      }
      
      
      # extract segmented
      segmented_ext <- segmented[extract_index,]
      # create grouping index
      group_index <- rep(NA, length(extract_index))
      n <- 1
      for (i in 1:(length(extract_index)-1)){
        if (extract_index[i] != extract_index[i+1] & (i + 1) != length(extract_index)){
          group_index[i] <- n
        } else if (extract_index[i] == extract_index[i+1]){
          group_index[i] <- n
          n <- n + 1
        } else if (i + 1 == length(extract_index)){
          group_index[i] <- n
          group_index[i+1] <- n
        }
      }
      segmented_ext <- as.data.frame(segmented_ext) %>% 
        mutate(group_index = group_index,
               SegID = x) %>% 
        select(SegID, group_index, X, Y, XY)
    } else {
      segmented_ext <- orig %>% 
        mutate(group_index = 1,
               SegID = x) %>% 
        select(SegID, group_index, X, Y, XY)
    }
    # # some cases has the last point duplicated in the same location. drop those. (segid 55839)
    # if (segmented_ext[nrow(segmented_ext)-1,'XY'] == segmented_ext[nrow(segmented_ext),'XY']){
    #   segmented_ext <- segmented_ext[1:(nrow(segmented_ext)-1),]
    # }
    
  }) %>% do.call('rbind', .)
  
  to_lines <- segmented_lines.list %>% 
    sf::st_as_sf(coords = c("X", "Y"), crs = 26967, dim = "XY") %>% 
    group_by(SegID, group_index) %>% 
    summarise(do_union = FALSE) %>% 
    sf::st_cast("LINESTRING") %>% 
    ungroup()
  
  return(to_lines)
}


# -- Calculate degrees -- # 
calculate_degree_endp = function(segments, clean_line = TRUE){
  # with an input line segment, generates two endpoints and calcualtes the
  # heading direction of those points
  
  # Arguements
  #   segments (sf object, "LINESTRING"): sf object of line shapefile representing street 
  #   centerline. This line should be already broken at all intersections.
  
  # Return
  #   sf "POINT" object with 'azimuth', "SegID", and "geometry" columns
  
  
  # # Delete duplicate segments
  if (clean_line == TRUE){
    segments = sf::st_difference(segments)
  }
  
  # Dataframe to receive all point information
  endp_all = data.frame(matrix(nrow = (nrow(segments)*2), ncol = 4))
  names(endp_all) = c("X", "Y", "azimuth", "SegID")
  
  # For loop through all line segments
  message("Looping through each segment")
  for (i in 1:nrow(segments)){
    # First line segment
    points = data.frame(sf::st_coordinates(segments[i,])[,c("X", "Y")])
    a = points[1:2,]
    a_degree = atan2(a[2,"X"] - a[1,"X"],
                     a[2,"Y"] - a[1,"Y"])*180/pi
    
    if (a_degree < 0){
      a_degree = a_degree + 360
    }
    
    # Last line segment
    z = points[(nrow(points)-1):nrow(points),]
    z_degree = atan2(z[2,"X"] - z[1,"X"],
                     z[2,"Y"] - z[1,"Y"])*180/pi
    
    if (z_degree < 0){
      z_degree = z_degree + 360
    }
    
    overall = points[c(1,nrow(points)),]
    overall_degree = atan2(overall[2,"X"] - overall[1,"X"],
                           overall[2,"Y"] - overall[1,"Y"])*180/pi
    
    if (overall_degree < 0){
      overall_degree = overall_degree + 360
    }
    
    # Assigning the direction to the endpoints
    a = cbind(a[1,], azimuth = a_degree)
    z = cbind(z[2,], azimuth = z_degree)
    az = rbind(a,z)
    az$SegID = as.numeric(sf::st_set_geometry(segments, NULL)[i,"SegID"])
    
    # Flipping the direction
    if (overall_degree > 0 & overall_degree <= 90){
      flip = which.max(az$X)
      az$azimuth[flip] = az$azimuth[flip] + 180
      
    } else if (overall_degree > 90 & overall_degree <= 180){
      flip = which.min(az$Y)
      az$azimuth[flip] = az$azimuth[flip] + 180
      
    } else if (overall_degree > 180 & overall_degree <= 270){
      flip = which.min(az$X)
      az$azimuth[flip] = az$azimuth[flip] - 180
    } else if (overall_degree > 270 & overall_degree <= 360){
      flip = which.max(az$Y)
      az$azimuth[flip] = az$azimuth[flip] - 180
    }
    
    # Appending it to endp_all
    endp_all[c(i*2-1,i*2),] = az
    if (i %% 1000 == 0){
      message(paste0("Finished ", i, " th segment."))
    }
  }
  
  # Back to sf
  endp_all = sf::st_as_sf(endp_all, coords = c("X", "Y"), dim = "XY", crs = sf::st_crs(segments)) 
  
  return(endp_all)
}

# -- Create dense points using calculate_degree_endp() -- # 
create_dense_points <- function(to_lines){
  message("[INFO] Creating dense points")
  # Create points using calculate_degree_endp()
  densepoints <- calculate_degree_endp(to_lines, clean_line = FALSE) %>% 
    # Extract the XY coordinates and label them dense
    mutate(x_coord = st_coordinates(.)[,1],
           y_coord = st_coordinates(.)[,2]) %>% 
    mutate(xy_coord = paste0(x_coord,',',y_coord),
           type = "dense") %>% 
    # Delete duplicates
    filter(!duplicated(.$xy_coord)) %>% 
    select(-x_coord, -y_coord, -xy_coord) %>% 
    # Insert sequence ID that starts from 2 for every new segment
    group_by(SegID) %>% 
    mutate(sequence_id = seq(2, (n()+1))) %>% 
    ungroup()
  
  return(densepoints)
}


# -- Create intersection points using calculate_degree_endp() -- #
create_inter_points <- function(input_segment){
  message("[INFO] Creating intersection points")
  # Create points using calculate_degree_endp()
  intersectionpoints <- calculate_degree_endp(input_segment, clean_line = FALSE) %>% 
    # Label them intersection
    mutate(type = "intersection") %>% 
    # Insert sequence ID as 1 and 1000
    group_by(SegID) %>% 
    mutate(sequence_id = c(1, 1000)) %>% 
    ungroup()
  
  return(intersectionpoints)
}

# -- Base function for getting GSV metadata to find out the exact location of images -- #
request_meta <- function(input_point, key){  
  base_url <- "https://maps.googleapis.com/maps/api/streetview/metadata?" 
  location <- input_point$xy_coord
  full_url <- paste0(base_url,"location=",location, "&key=", key)
  request <- httr::GET(url = full_url)
  response <- httr::content(request, as = "text", encoding = "UTF-8")
  df <- jsonlite::fromJSON(response, flatten = TRUE) %>% 
    data.frame()
  return(df)
}

# -- Get GSV metadata
collect_metadata <- function(input_point, key){
  for (i in 1:nrow(input_point)){
    # Get querry for this iteration
    response <- request_meta(input_point[i,], key = key)
    # Iteration Tracker
    if (i %% 50 == 0) print(paste0("[INFO] working on ", i, "th iteration"))
    
    # If first iteration..
    if (i == 1){
      metadata <- cbind(response, input_point[i,] %>% st_set_geometry(NULL))
    } else {
      # If GSV image is missing..
      if (response$status != "OK"){
        metadata <- rbind(metadata,
                          cbind(data.frame("copyright" = NA,
                                           "date" = NA,
                                           "location.lat" = strsplit(input_point$xy_coord[i], ",")[[1]][1],
                                           "location.lng" = strsplit(input_point$xy_coord[i], ",")[[1]][2],
                                           "pano_id" = NA,
                                           "status" = response$status),
                                input_point[i,] %>% st_set_geometry(NULL)))
        # If GSV image exists..
      } else {
        metadata <- rbind(metadata, cbind(response, input_point[i,] %>% st_set_geometry(NULL)))
      }
    }
  }
  
  return(metadata)
}

# -- Post-process GSV metadata -- #
relocate_images <- function(metadata, potential_points){
  # Using the metadata information from Google, 
  # this function (1) creates image points at the exact image location and
  # (2) joins the metadata with potential_points, 
  # (3) adjusts the azimuth, and 
  # (4) deletes overlapping points
  
  # Arguements
  #   metadata (data.frame): Return from GSV metadata API call.
  #   potential_points(sf object, "POINT"): Result of rbind of densepoints and intersectionpoints
  
  # Return
  #   sf "POINT" with date, coordinates, pano_id, and status variables
  
  # (1) Creates points at the exact image location
  metadata_shp <- metadata %>% 
    mutate(gsv_snap_xcoord = location.lng,
           gsv_snap_ycoord = location.lat) %>% 
    st_as_sf(coords = c('location.lng', 'location.lat'), dim = 'XY', crs = 4326)
  
  # (2) Joins the metadata with potential_points
  audit_point_noAzi <- metadata_shp %>% 
    select(date, pano_id, status, point_id, gsv_snap_xcoord, gsv_snap_ycoord) %>% 
    left_join(potential_points %>% st_set_geometry(NULL), by = 'point_id')
  
  # (3) Adjusts the azimuth & delete overlapping points
  # Step 1 - Rotate 90 degrees
  audit_point_wAzi <- audit_point_noAzi %>% 
    mutate(azimuth = case_when(
      type == "dense" ~ azimuth + 90,
      type == 'intersection' ~ azimuth
    )) %>% 
    mutate(azimuth = case_when(
      azimuth > 360 ~ azimuth - 360,
      azimuth < 0 ~ azimuth + 360,
      TRUE ~ azimuth
    ))
  
  # Step 2 - delete overlapping Dense points
  audit_intersection <- audit_point_wAzi %>% 
    filter(type == 'intersection') %>% 
    mutate(point_id = as.character(point_id)) %>% 
    select(pano_id, date, status, SegID, type, azimuth, point_id, sequence_id)
  
  audit_dense <- audit_point_wAzi %>% 
    filter(type == 'dense', !is.na(pano_id)) %>% 
    group_by(pano_id, date, status, SegID, type) %>% 
    summarise(azimuth = mean(azimuth),
              point_id = paste(point_id, collapse = "-"),
              sequence_id = mean(sequence_id)) %>% 
    ungroup()
  
  audit_point_wAzi <- audit_intersection %>% 
    bind_rows(audit_dense) 
  
  
  # Step 3 - Flipping Densepoints
  audit_point_2 <- audit_point_wAzi %>% 
    filter(type == 'dense') %>% 
    mutate(azimuth = case_when(
      azimuth > 180 ~ azimuth - 180,
      azimuth <=180 ~ azimuth + 180
    )) %>% 
    mutate(point_id = paste(point_id, "flip", sep = "-"))
  
  # Combine Step 2 and Step 3
  audit_point_wAzi <- audit_point_wAzi %>% 
    bind_rows(audit_point_2) %>% 
    arrange(SegID)
  
  # Step 4
  intersection_pano_id <- audit_point_wAzi %>% 
    filter(type == 'intersection',
           !is.na(pano_id)) %>% 
    with(unique(.$pano_id))
  
  densepoint_index <- audit_point_wAzi$type == 'dense'
  interpano_index <- audit_point_wAzi$pano_id %in% intersection_pano_id
  densepoint_drop_index <- as.logical(densepoint_index * interpano_index)
  
  audit_point_wAzi <- audit_point_wAzi %>% 
    filter(!densepoint_drop_index) 
  
  return(audit_point_wAzi)
}

# -- Tell apart Right and Left points -- #
tell_apart_left_right <- function(input_point, seg_azimuth){
  # Using the point_id variable and whether it contains the word "flip" or not,
  # this function determines the 'side' of the point it is looking at from the
  # perspective of the starting point of the given segment
  
  # Arguements
  #   input_point (sf object, "POINT"): Point shapefile that contains points that need to be
  #                                     identified in terms of L or R
  #   seg_azimuth (data.frame): A data.frame that contains SegID and SegAzi
  
  # Return
  #   sf "POINT" with added variable called LR
  
  # Tell whether it is R or L
  audit_point <- input_point %>% 
    left_join(seg_azimuth, by = 'SegID') %>% 
    group_by(SegID) %>% 
    arrange(SegID, type, sequence_id) %>% 
    mutate(LR = case_when(
      type == 'dense' & grepl("flip", point_id) == TRUE ~ "R",
      type == 'dense' & grepl("flip", point_id) == FALSE ~ "L",
      type == 'intersection' ~ "I"
    )) %>% 
    ungroup()
  
  # Reorder Right side
  for (segid in unique(audit_point$SegID)){
    # Extract SegID and Right side
    extract <- audit_point %>% filter(SegID == segid, LR == "R")
    seqid <- extract$sequence_id
    
    # Sort seqid
    seqid <- sort(seqid, decreasing = T)
    audit_point$sequence_id[audit_point$SegID == segid & audit_point$LR == 'R'] <- seqid
  }   
  
  return(audit_point)
}

# -- Duplicate the intersection points and change their azimuth by -45 and +45 -- #
duplicate_inter_points <- function(input_point){
  # Because there are two intersection points with +45 and -45,
  # this function duplicates the intersection point and adjust the azimuth accordingly.
  
  # Arguements
  #   input_point (sf object, "POINT"): Point shapefile that contains audit points.
  
  # Return
  #   sf "POINT" with added rows for two intersection points for one intersection.
  
  # Loop through the data.frame, find intersection points, duplicate them, and adjust the azimuth
  for (i in 1:nrow(input_point)){
    if (input_point$type[i] == "dense") {next}
    else if (input_point$type[i] == "intersection"){
      original_azimuth <- input_point$azimuth[i]
      
      # rotate the original intersection point
      input_point$azimuth[i] <- input_point$azimuth[i] - 45
      if (input_point$azimuth[i] < 0) {input_point$azimuth[i] <- input_point$azimuth[i] + 360}
      
      # Dealing with the second intersection point
      temp <- input_point[i,]
      temp$azimuth <- original_azimuth + 45
      if (temp$azimuth > 360) {temp$azimuth <- temp$azimuth - 360}
      
      # Collect the temps in a separate data.frame. This data.frame will be appended back to the original data.frame.
      if (!exists("temp_collector")) {temp_collector <- temp}
      else if (exists("temp_collector")) {temp_collector <- bind_rows(temp_collector, temp)}
    }
  }
  
  # Row-bind it back to the original data.frame
  input_point <- input_point %>% 
    bind_rows(temp_collector) %>% 
    arrange(SegID, type, point_id)
  
  return(input_point)
}

# -- Add points for Densepoints for TOP images
add_top_points <- function(input_point){
  # Because there are two intersection points with +45 and -45,
  # this function duplicates the intersection point and adjust the azimuth accordingly.
  
  # Arguements
  #   input_point (sf object, "POINT"): Point shapefile that contains audit points.
  
  # Return
  #   sf "POINT" with added rows for two intersection points for one intersection.
  
  # Loop through the data.frame, find intersection points, duplicate them, and adjust the azimuth
  audit_point.top <- input_point %>% 
    filter(LR == "L") %>% 
    mutate(LR = "T")
  
  audit_point <- input_point %>% 
    bind_rows(audit_point.top) %>% 
    arrange(SegID, type, LR, sequence_id)
  
  return(audit_point)
}


# ------------------ Main Function ------------------ # 
prepare_download_points <- function(input_shape, state, county, year = 2018, key = Sys.getenv("google_api")){
  
  if ((missing(state) & missing(county)) & !missing(input_shape)) {
    # Rename input_shape as rd
    rd <- input_shape %>% st_transform(crs = 4326)
    
  } else if ((missing(state) | missing(county)) & missing(input_shape)) {
    message("[ERROR] Either a state-county pair or a road shapefile is required")
    break
    
  } else if ((!missing(state) | !missing(county)) & !missing(input_shape)) {
    message("[ERROR] Provide either one of the state-county pair or the road shapefile, but not both")
    break
    
  } else {
    # -- Get TIGER road centerline shapefile -- #
    rd <- roads(state = state, county = county, year = year)
  }
  
  # -- Self-split the road shapefile at intersections -- #
  intersections <- create_intersection(rd, include_deadend = FALSE)
  edge <- break_line_at_point(rd, intersections, clean_line = TRUE)
  edge_by_5m <- break_seg_at_interval(edge, interval = 5)
  
  # -- Get SegAzi for later
  seg_azimuth <- get_segAzi(edge)
  
  # -- Create densepoints & intersection points that contain Azimuth for each point -- #
  dense_point <- create_dense_points(edge_by_5m)
  inter_point <- create_inter_points(edge)
  message("All points are created")
  
  # -- Rbind dense & intersection points
  # Identifying from densepoints that are overlapping with endpoints
  overlapping_points <- dense_point %>% 
    st_intersects(inter_point %>% st_buffer(units::set_units(10, "meter")))
  
  # To logical
  overlapping_points_logical <- lengths(overlapping_points) > 0
  
  # Drop dense points that are within 10 meters from intersection points
  dense_point <- dense_point[!overlapping_points_logical,]
  
  # Binding rows
  potential_points <- rbind(dense_point, inter_point) %>% 
    mutate(point_id = seq(1:nrow(.)))
  
  # Add xy_coord columns
  potential_points <- potential_points %>% 
    st_transform(crs = 4326) %>% 
    mutate(x_coord = st_coordinates(.)[,1],
           y_coord = st_coordinates(.)[,2]) %>% 
    mutate(xy_coord = paste0(y_coord, ",", x_coord))
  
  # -- Get per-segment azimuth for later use
  seg_azimuth <- get_segAzi(edge)
  
  # -- Check metadata and relocate image locations to exact image locations -- # 
  metadata <- collect_metadata(potential_points, key = key)
  metadata_shp <- relocate_images(metadata, potential_points)
  
  # -- Add LR variable, duplicate intersections, and add TOP images -- # 
  audit_point_LR <- tell_apart_left_right(metadata_shp, seg_azimuth)
  audit_point_double_inter <- duplicate_inter_points(audit_point_LR)
  audit_point <- add_top_points(audit_point_double_inter)
  
  # -- Add y_coord and x_coord
  audit_point <- audit_point %>% 
    mutate(x_coord = st_coordinates(.)[,1],
           y_coord = st_coordinates(.)[,2])
  
  # -- Return
  return(
    list("rd" = rd, 
              "intersections" = intersections, 
              "edge" = edge,
              "edge_by_5m" = edge_by_5m, 
              "dense_point" = dense_point,
              "inter_point" = inter_point,
              "potential_points" = potential_points,
              "audit_points" = audit_point)
    )
}
