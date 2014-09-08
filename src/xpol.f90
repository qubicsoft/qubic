module xpol

  implicit none
  integer, parameter :: dp = kind(1.d0)

  interface
    subroutine wig3j(l2, l3, m2, m3, l1min, l1max, array, ndim, ier)
      real*8, intent(in)  :: l2, l3, m2, m3
      real*8, intent(out) :: l1min, l1max
      integer, intent(in) :: ndim
      real*8, dimension(*), intent(out) :: array
      integer :: ier
    end subroutine wig3j
  end interface

contains

  subroutine Mll_blocks_pol(lmax, well, nwell, mll_TT_TT, mll_EE_EE, mll_EE_BB,&
                     mll_TE_TE, mll_EB_EB, ier)
    integer, intent(in)  :: lmax
    real(dp), intent(in) :: well(nwell)
    integer, intent(in)  :: nwell
    real(dp), intent(out), dimension(0:lmax, 0:lmax) ::                        &
         mll_TT_TT, mll_EE_EE, mll_EE_BB, mll_TE_TE, mll_EB_EB
    integer, intent(out) :: ier

    real(dp), parameter  :: one_fourpi = 1.d0 / (atan(1.d0) * 16.d0)
    real(dp), dimension(2*lmax+1)  :: wigner0, wigner2
    real(dp), dimension(0:nwell-1) :: well_TT, well_TP, well_PP, thewell, ell
    real(dp) :: rl1, rl2, rl1min, rl1max, wig00, wig02, wig22, coef,           &
                sum_TT, sum_TE, sum_EE_EE, sum_EE_BB, sum_EB
    integer  :: l, l1, l2, l1min, l1max, ndim, ier_, iwig
    logical  :: mask

    ell = [(l, l=0, nwell-1)]
    thewell = well * (2 * ell + 1)
    well_TT = thewell
    well_TP = thewell
    well_PP = thewell
    
    ier = 0
    !$omp parallel do default(private) &
    !$omp& shared(ier, lmax, nwell, well_tt, well_tp, well_pp) &
    !$omp& shared(mll_TT_TT, mll_EE_EE, mll_EE_BB, mll_TE_TE, mll_EB_EB)
    do l1=0, lmax
      rl1 = real(l1, dp)
      do l2=0, lmax
        rl2 = real(l2, dp)
        ndim = l1 + l2 + 1
        call wig3j(rl1, rl2, 0.d0, 0.d0, rl1min, rl1max, wigner0, ndim, ier_)
        if (ier_ /= 0) then
            ier = ier_
            cycle
        end if
        l1min = int(rl1min)
        l1max = min(int(rl1max), nwell - 1)

        if (l1 < 2 .or. l2 < 2) then
            ! no need to get rl1min & rl1max again
            wigner2(:l1max - l1min + 1) = 0.d0
        else
            call wig3j(rl1, rl2, -2.d0, 2.d0, rl1min, rl1max, wigner2, ndim,   &
                       ier_)
            if (ier_ /= 0) then
                ier = ier_
                cycle
            end if
        end if

        sum_TT = 0.d0
        sum_TE = 0.d0
        sum_EE_EE = 0.d0
        sum_EE_BB = 0.d0
        mask = mod(l1 + l2 + l1min, 2) == 0
        do l = l1min, l1max
          iwig = l - l1min + 1
          wig00 = wigner0(iwig) * wigner0(iwig)
          wig02 = wigner0(iwig) * wigner2(iwig)
          wig22 = wigner2(iwig) * wigner2(iwig)
          sum_TT = sum_TT + well_TT(l) * wig00
          if (mask) sum_TE = sum_TE + well_TP(l) * wig02
          if (mask) sum_EE_EE = sum_EE_EE + well_PP(l) * wig22
          if (.not. mask) sum_EE_BB = sum_EE_BB + well_PP(l) * wig22
          mask = .not. mask
        end do
        sum_EB = sum_EE_EE + sum_EE_BB

        coef = (2 * l2 + 1) * one_fourpi
        mll_TT_TT(l1, l2) = coef * sum_TT
        mll_EE_EE(l1, l2) = coef * sum_EE_EE
        mll_EE_BB(l1, l2) = coef * sum_EE_BB
        mll_TE_TE(l1, l2) = coef * sum_TE
        mll_EB_EB(l1, l2) = coef * sum_EB

      end do
    end do
    !$omp end parallel do

  end subroutine Mll_blocks_pol

end module xpol
