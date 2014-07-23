! ***********************************************************************
!
!   Copyright (C) 2011  Bill Paxton
!
!   this file is part of mesa.
!
!   mesa is free software; you can redistribute it and/or modify
!   it under the terms of the gnu general library public license as published
!   by the free software foundation; either version 2 of the license, or
!   (at your option) any later version.
!
!   mesa is distributed in the hope that it will be useful, 
!   but without any warranty; without even the implied warranty of
!   merchantability or fitness for a particular purpose.  see the
!   gnu library general public license for more details.
!
!   you should have received a copy of the gnu library general public license
!   along with this software; if not, write to the free software
!   foundation, inc., 59 temple place, suite 330, boston, ma 02111-1307 usa
!
! ***********************************************************************
 
      module run_star_extras

      use star_lib
      use star_def
      use const_def
      
      implicit none
      
      integer :: time0, time1, clock_rate
      real(dp), parameter :: expected_runtime = 0.1 ! minutes


      contains
      
      
      subroutine extras_controls(s, ierr)
         type (star_info), pointer :: s
         integer, intent(out) :: ierr
         ierr = 0
         
         s% other_atm => tutorial_other_atm
      end subroutine extras_controls
      
      
      integer function extras_startup(s, id, restart, ierr)
         type (star_info), pointer :: s
         integer, intent(in) :: id
         logical, intent(in) :: restart
         integer, intent(out) :: ierr
         ierr = 0
         extras_startup = 0
         call system_clock(time0,clock_rate)
         if (.not. restart) then
            call alloc_extra_info(s)
         else ! it is a restart
            call unpack_extra_info(s)
         end if
      end function extras_startup
      
      
      subroutine extras_after_evolve(s, id, id_extra, ierr)
         type (star_info), pointer :: s
         integer, intent(in) :: id, id_extra
         integer, intent(out) :: ierr
         real(dp) :: dt
         ierr = 0
         call system_clock(time1,clock_rate)
         dt = dble(time1 - time0) / clock_rate / 60
         if (dt > 10*expected_runtime) then
            write(*,'(/,a30,2f12.1,a,/)') '>>>>>>> EXCESSIVE runtime', &
               dt, expected_runtime, '   <<<<<<<<<  ERROR'
         else
            write(*,'(/,a50,2f12.1,99i10/)') 'runtime, prev time, retries, backups, steps', &
               dt, expected_runtime, s% num_retries, s% num_backups, s% model_number
         end if
      end subroutine extras_after_evolve
      

      ! returns either keep_going, retry, backup, or terminate.
      integer function extras_check_model(s, id, id_extra)
         type (star_info), pointer :: s
         integer, intent(in) :: id, id_extra
         extras_check_model = keep_going         
      end function extras_check_model


      integer function how_many_extra_history_columns(s, id, id_extra)
         type (star_info), pointer :: s
         integer, intent(in) :: id, id_extra
         how_many_extra_history_columns = 0
      end function how_many_extra_history_columns
      
      
      subroutine data_for_extra_history_columns(s, id, id_extra, n, names, vals, ierr)
         type (star_info), pointer :: s
         integer, intent(in) :: id, id_extra, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n)
         integer, intent(out) :: ierr
         ierr = 0
      end subroutine data_for_extra_history_columns

      
      integer function how_many_extra_profile_columns(s, id, id_extra)
         type (star_info), pointer :: s
         integer, intent(in) :: id, id_extra
         how_many_extra_profile_columns = 0
      end function how_many_extra_profile_columns
      
      
      subroutine data_for_extra_profile_columns(s, id, id_extra, n, nz, names, vals, ierr)
         type (star_info), pointer :: s
         integer, intent(in) :: id, id_extra, n, nz
         character (len=maxlen_profile_column_name) :: names(n)
         real(dp) :: vals(nz,n)
         integer, intent(out) :: ierr
         integer :: k
         ierr = 0
      end subroutine data_for_extra_profile_columns
      

      ! returns either keep_going or terminate.
      integer function extras_finish_step(s, id, id_extra)
         type (star_info), pointer :: s
         integer, intent(in) :: id, id_extra
         integer :: ierr
         extras_finish_step = keep_going
         call store_extra_info(s)
      end function extras_finish_step
      
      
      ! routines for saving and restoring extra data so can do restarts
         
         ! put these defs at the top and delete from the following routines
         !integer, parameter :: extra_info_alloc = 1
         !integer, parameter :: extra_info_get = 2
         !integer, parameter :: extra_info_put = 3
      
      
      subroutine alloc_extra_info(s)
         integer, parameter :: extra_info_alloc = 1
         type (star_info), pointer :: s
         call move_extra_info(s,extra_info_alloc)
      end subroutine alloc_extra_info
      
      
      subroutine unpack_extra_info(s)
         integer, parameter :: extra_info_get = 2
         type (star_info), pointer :: s
         call move_extra_info(s,extra_info_get)
      end subroutine unpack_extra_info
      
      
      subroutine store_extra_info(s)
         integer, parameter :: extra_info_put = 3
         type (star_info), pointer :: s
         call move_extra_info(s,extra_info_put)
      end subroutine store_extra_info
      
      
      subroutine move_extra_info(s,op)
         integer, parameter :: extra_info_alloc = 1
         integer, parameter :: extra_info_get = 2
         integer, parameter :: extra_info_put = 3
         type (star_info), pointer :: s
         integer, intent(in) :: op
         
         integer :: i, j, num_ints, num_dbls, ierr
         
         i = 0
         ! call move_int or move_flg    
         num_ints = i
         
         i = 0
         ! call move_dbl       
         
         num_dbls = i
         
         if (op /= extra_info_alloc) return
         if (num_ints == 0 .and. num_dbls == 0) return
         
         ierr = 0
         call star_alloc_extras(s% id, num_ints, num_dbls, ierr)
         if (ierr /= 0) then
            write(*,*) 'failed in star_alloc_extras'
            write(*,*) 'alloc_extras num_ints', num_ints
            write(*,*) 'alloc_extras num_dbls', num_dbls
            stop 1
         end if
         
         contains
         
         subroutine move_dbl(dbl)
            real(dp) :: dbl
            i = i+1
            select case (op)
            case (extra_info_get)
               dbl = s% extra_work(i)
            case (extra_info_put)
               s% extra_work(i) = dbl
            end select
         end subroutine move_dbl
         
         subroutine move_int(int)
            integer :: int
            i = i+1
            select case (op)
            case (extra_info_get)
               int = s% extra_iwork(i)
            case (extra_info_put)
               s% extra_iwork(i) = int
            end select
         end subroutine move_int
         
         subroutine move_flg(flg)
            logical :: flg
            i = i+1
            select case (op)
            case (extra_info_get)
               flg = (s% extra_iwork(i) /= 0)
            case (extra_info_put)
               if (flg) then
                  s% extra_iwork(i) = 1
               else
                  s% extra_iwork(i) = 0
               end if
            end select
         end subroutine move_flg
      
      end subroutine move_extra_info
      
      
      subroutine tutorial_other_atm( &
            id, M, R, L, X, Z, kap, Teff, &
            lnT, dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap, &
            lnP, dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap, &
            which_atm_option, switch_to_grey_as_backup, ierr)
         integer, intent(in) :: id ! star id if available; 0 otherwise
         ! cgs units
         real(dp), intent(in) :: M ! enclosed mass at photosphere
         real(dp), intent(in) :: R ! radius at photosphere
         real(dp), intent(in) :: L ! luminosity at photosphere
         real(dp), intent(in) :: X ! hydrogen mass fraction
         real(dp), intent(in) :: Z ! metallicity
         real(dp), intent(in) :: kap 
            ! opacity above photosphere (average by mass)
            ! only used for grey atm approximation.
         
         real(dp), intent(out) :: Teff ! temperature at photosphere
         real(dp), intent(out) :: lnT ! natural log of temperature at base of atmosphere
         real(dp), intent(out) :: lnP ! natural log of pressure at base of atmosphere (Pgas + Prad)
         
         ! partial derivatives of lnT and lnP
         real(dp), intent(out) :: dlnT_dL, dlnT_dlnR, dlnT_dlnM, dlnT_dlnkap
         real(dp), intent(out) :: dlnP_dL, dlnP_dlnR, dlnP_dlnM, dlnP_dlnkap
         
         integer, intent(in) :: which_atm_option
            ! atm_simple_photosphere or one of the options for tables
            ! T(tau) integration is not supported in this routine -- call atm_int_T_tau instead
         logical, intent(in) :: switch_to_grey_as_backup
            ! if you select a table option, but the args are out of the range of the tables,
            ! then this flag determines whether you get an error or the code automatically
            ! switches to option = atm_simple_photosphere as a backup.

         integer, intent(out) :: ierr  ! == 0 means AOK
         
         Teff = 44.7; lnT = 3.8; lnP = -2.7
         dlnT_dL = 0; dlnT_dlnR = 0; dlnT_dlnM = 0; dlnT_dlnkap = 0
         dlnP_dL = 0; dlnP_dlnR = 0; dlnP_dlnM = 0; dlnP_dlnkap = 0
         
         !write(*,*) 'no implementation for other_atm'
         !ierr = -1

      end subroutine tutorial_other_atm

      end module run_star_extras
      
